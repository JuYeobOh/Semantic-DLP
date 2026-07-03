"""run_experiment() — 등록·탐지 파이프라인 한 번 실행 + 계측/저장.

흐름: split → query sets → (임베딩 캐시)임베딩 → 인덱스 → 검색 → 투표 → 판별 → 평가.
★ Latency 규칙(§4): 인덱스 빌드 시간은 추론 latency 에서 제외.
   inference_total_sec = 쿼리 임베딩 + 검색 + 투표.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdlp.chunking.core import build_chunks_df
from sdlp.embedding.cache import (
    embedding_cache_dir,
    embedding_cache_exists,
    load_embedding_artifact,
    load_embedding_config,
    save_embedding_artifact,
)
from sdlp.embedding.st import STTextEmbedder
from sdlp.index.faiss_hnsw import build_faiss_hnsw_index
from sdlp.instrument import index_size_mb, throughput, timer
from sdlp.io import ensure_dir, load_prepared_set, save_json, save_parquet
from sdlp.metrics.core import (
    build_errors_by_family,
    build_false_positive_pairs,
    build_query_manifest,
    evaluate_run,
    save_metrics_json,
)
from sdlp.pipeline.config import RunConfig
from sdlp.retrieval.core import search_index
from sdlp.schemas import CHUNK_META_COLUMNS
from sdlp.splits.core import build_run_query_sets, make_split
from sdlp.voting.core import vote_by_document


# 임베딩 단계 산출물 묶음 (ref/query 청크·임베딩 + 단계별 소요시간).
@dataclass
class _EmbedResult:
    ref_chunks: pd.DataFrame
    ref_emb: np.ndarray
    query_chunks: pd.DataFrame
    query_emb: np.ndarray
    chunk_sec: float
    embed_ref_sec: float
    embed_query_sec: float
    embed_source: str


# 한 prepared set 의 청크 임베딩을 캐시에서 가져오거나 새로 빌드.
# 반환: (chunks_df, embeddings, chunk_sec, embed_sec, source). source ∈ {"cached","live"}.
def _get_or_embed(cfg: RunConfig, set_name: str, embedder: STTextEmbedder):
    cache_dir = embedding_cache_dir(cfg.artifacts_dir, cfg.embed_spec, cfg.chunk_spec.slug(), set_name)
    if embedding_cache_exists(cache_dir):
        chunks_df, emb = load_embedding_artifact(cache_dir)
        c = load_embedding_config(cache_dir)
        return chunks_df, emb, float(c["chunk_sec"]), float(c["embed_sec"]), "cached"

    full = load_prepared_set(cfg.prepared_dir, set_name)
    tokenizer = embedder.tokenizer if cfg.chunk_spec.mode == "token" else None
    with timer() as tc:
        chunks_df = build_chunks_df(full, cfg.chunk_spec, tokenizer=tokenizer)
    emb, embed_sec = embedder.encode_chunks_df(chunks_df)
    save_embedding_artifact(cache_dir, chunks_df, emb, cfg.embed_spec,
                            chunk_sec=tc.sec, embed_sec=embed_sec, n_docs=int(len(full)))
    return chunks_df, emb, tc.sec, embed_sec, "live"


# chunks_df/embeddings 를 주어진 doc_id 집합으로 필터.
def _filter(chunks_df: pd.DataFrame, emb: np.ndarray, doc_ids: set[str]):
    mask = chunks_df["doc_id"].isin(doc_ids).to_numpy()
    return chunks_df.loc[mask].reset_index(drop=True), emb[mask]


# 여러 source 를 하나로 요약.
def _combine(sources: list[str]) -> str:
    s = set(sources)
    return "cached" if s == {"cached"} else ("live" if s == {"live"} else "mixed")


# split → reference/positive/benign query sets + 정답 manifest (split/manifest 저장).
def _prepare_run(cfg: RunConfig, run_dir: Path):
    split = make_split(cfg.prepared_dir, cfg.splits_dir, cfg.dataset,
                       cfg.resolved_original_set, cfg.split_seed, cfg.split_ratio)
    save_parquet(split, run_dir / "splits_used.parquet")
    qs = build_run_query_sets(cfg.prepared_dir, split, cfg.resolved_original_set,
                              cfg.resolved_variant_sets, cfg.include_original_as_positive)
    manifest = build_query_manifest(qs.positive_df, qs.benign_df)
    save_parquet(manifest, run_dir / "query_manifest.parquet")
    return qs, manifest


# reference(기밀 원본) + query(원본 query doc + 각 변형 set) 청크 임베딩 (캐시 사용). ref_chunks 저장.
def _embed_ref_and_query(cfg: RunConfig, qs, run_dir: Path) -> _EmbedResult:
    embedder = STTextEmbedder(cfg.embed_spec)
    chunk_sec = embed_ref_sec = embed_query_sec = 0.0
    sources: list[str] = []

    orig_chunks, orig_emb, c_s, e_s, src = _get_or_embed(cfg, cfg.resolved_original_set, embedder)
    chunk_sec += c_s; embed_ref_sec += e_s; sources.append(src)
    ref_chunks, ref_emb = _filter(orig_chunks, orig_emb, set(qs.reference_df["doc_id"]))
    save_parquet(ref_chunks, run_dir / "ref_chunks.parquet")

    query_ids = set(qs.positive_df["doc_id"]) | set(qs.benign_df["doc_id"])
    parts_c: list[pd.DataFrame] = []
    parts_e: list[np.ndarray] = []
    oc, oe = _filter(orig_chunks, orig_emb, query_ids)
    if len(oc):
        parts_c.append(oc); parts_e.append(oe)
    for vset in cfg.resolved_variant_sets:
        v_chunks, v_emb, c_s, e_s, src = _get_or_embed(cfg, vset, embedder)
        chunk_sec += c_s; embed_query_sec += e_s; sources.append(src)
        vc, ve = _filter(v_chunks, v_emb, query_ids)
        if len(vc):
            parts_c.append(vc); parts_e.append(ve)

    if parts_c:
        query_chunks = pd.concat(parts_c, ignore_index=True)
        query_emb = np.concatenate(parts_e, axis=0)
    else:
        query_chunks = ref_chunks.iloc[0:0].copy()
        query_emb = np.zeros((0, ref_emb.shape[1]), dtype=np.float32)

    return _EmbedResult(ref_chunks, ref_emb, query_chunks, query_emb,
                        chunk_sec, embed_ref_sec, embed_query_sec, _combine(sources))


# timing/counts/throughput/index 를 metrics 에 채우고 metrics.json 저장 (§4 latency 규칙).
def _finalize_metrics(metrics, run_dir, qs, emb: _EmbedResult, votes,
                      index_build_sec, retrieval_sec, voting_sec, total_sec) -> None:
    inference_total = emb.embed_query_sec + retrieval_sec + voting_sec
    counts = {
        "n_ref_docs": int(qs.reference_df["doc_id"].nunique()),
        "n_ref_chunks": int(len(emb.ref_chunks)),
        "n_query_docs": int(len(qs.positive_df) + len(qs.benign_df)),
        "n_query_chunks": int(len(emb.query_chunks)),
        "n_queries": int(votes["query_doc_id"].nunique()) if len(votes) else 0,
    }
    metrics["timing_sec"] = {
        "chunk_sec": emb.chunk_sec,
        "embed_ref_sec": emb.embed_ref_sec,
        "embed_query_sec": emb.embed_query_sec,
        "embed_source": emb.embed_source,
        "index_build_sec": index_build_sec,      # 참고용, 추론 latency 제외
        "retrieval_sec": retrieval_sec,
        "voting_sec": voting_sec,
        "inference_total_sec": inference_total,
        "total_sec": total_sec,
    }
    metrics["counts"] = counts
    metrics["throughput"] = {
        "queries_per_sec": throughput(counts["n_queries"], inference_total),
        "query_chunks_embed_per_sec": throughput(counts["n_query_chunks"], emb.embed_query_sec),
        "query_chunks_search_per_sec": throughput(counts["n_query_chunks"], retrieval_sec),
    }
    metrics["index"] = {
        "size_mb": index_size_mb(run_dir / "index"),
        "n_vectors": int(len(emb.ref_chunks)),
        "dim": int(emb.ref_emb.shape[1]) if emb.ref_emb.ndim == 2 else 0,
    }
    save_metrics_json(metrics, run_dir / "metrics.json")


# 파이프라인 1회 실행. metrics dict 반환 + run_dir 에 산출물 저장.
def run_experiment(cfg: RunConfig) -> dict:
    run_dir = ensure_dir(Path(cfg.artifacts_dir) / "runs" / cfg.run_slug())
    save_json(cfg.as_serializable(), run_dir / "config.json")

    with timer() as t_total:
        qs, manifest = _prepare_run(cfg, run_dir)
        emb = _embed_ref_and_query(cfg, qs, run_dir)

        with timer() as t_idx:   # 인덱스 빌드 (추론 latency 제외)
            index = build_faiss_hnsw_index(emb.ref_emb, emb.ref_chunks[CHUNK_META_COLUMNS], cfg.faiss_config)
        index.save(run_dir / "index")

        with timer() as t_ret:
            retrieval = search_index(index, emb.query_chunks, emb.query_emb, cfg.top_k)
        if cfg.save_retrieval:
            save_parquet(retrieval, run_dir / "retrieval_topk.parquet")

        with timer() as t_vote:
            votes = vote_by_document(retrieval, use_score_weight=cfg.use_score_weight)
        save_parquet(votes, run_dir / "votes.parquet")

        metrics, eval_df = evaluate_run(manifest, votes, cfg.confidence_threshold)
        save_parquet(eval_df, run_dir / "per_query_eval.parquet")
        save_parquet(build_errors_by_family(eval_df), run_dir / "errors_by_family.parquet")
        save_parquet(build_false_positive_pairs(eval_df), run_dir / "fp_pairs.parquet")

    _finalize_metrics(metrics, run_dir, qs, emb, votes,
                      index_build_sec=t_idx.sec, retrieval_sec=t_ret.sec,
                      voting_sec=t_vote.sec, total_sec=t_total.sec)
    return metrics
