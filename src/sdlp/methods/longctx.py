"""embedding_longctx — 문서 통짜(no chunking) long-context 임베딩 top-1 매칭.

각 문서를 single 벡터로 인코딩 → query×ref cosine → 문서/family top-1.
**confidence = best-family cosine**(raw; threshold sweep·AUC 로 평가). 이름을 confidence 로 둬야
chunk_voting 과 동일한 evaluate_run/apply_threshold 경로에 drop-in 된다(값은 cosine).
기본 모델: ibm-granite/granite-embedding-97m-multilingual-r2 (encoder bi-encoder, context 32768, dim 384, 경량).
성능형 대안: ibm-granite/granite-embedding-311m-multilingual-r2 (dim 768, Matryoshka).

max_seq_length 초과 문서 처리(long_doc):
  "truncate"  = head 절단(앞 N 토큰 보존, 뒤 유실) — 기존 방식
  "mean_pool" = max_seq_length 윈도우로 쪼개 각각 인코딩 후 평균+정규화 (전문 보존)
  "exclude"   = 초과 문서 제외 (실험에서 빠짐)

prepared-set×long_doc 단위 임베딩 캐시(artifacts/cache/embeddings_longctx/{model}/{set}__{long_doc}.npy).
torch/sentence_transformers 는 지연 import. votes_df 는 FINAL VOTES 스키마(confidence) 호환.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sdlp.embedding.spec import model_slug

DEFAULT_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
DEFAULT_MAX_SEQ = 32768   # granite r2 context (97m/311m 모두 32k)
LONG_DOC_MODES = ("truncate", "mean_pool", "exclude")

# 프로세스 내 모델 재사용 (같은 설정이면 1회 로드).
_MODEL_CACHE: dict[str, object] = {}


# SentenceTransformer 로드 (지연 import, 프로세스 캐시). device None → cuda 있으면 cuda.
def _get_encoder(model_name: str, max_seq_length: int, device: str | None, dtype: str):
    import torch
    from sentence_transformers import SentenceTransformer

    key = f"{model_name}|{device}|{dtype}|{max_seq_length}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 초장문(32k) attention 을 math 커널(N×N 행렬 materialize)로 돌리면 OOM →
    # mem-efficient/flash SDPA 강제, math 끔. (짧은 문서엔 영향 없음.)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"torch_dtype": torch_dtype}, trust_remote_code=True)
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.truncation_side = "right"   # head truncation(앞 N 토큰 보존)
    model.max_seq_length = int(max_seq_length)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


# 문서 텍스트를 doc-level 벡터로 인코딩 (long_doc 전략 적용).
# 반환: (emb(N,D), 유지된 doc_ids, family_ids, n_overflow). exclude 는 유지 문서만, 나머지는 전체.
def _encode_docs(model, texts, doc_ids, family_ids, max_seq_length, batch_size, long_doc):
    def _enc(seg_texts, normalize):
        if not seg_texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), np.float32)
        return model.encode(seg_texts, batch_size=batch_size, normalize_embeddings=normalize,
                            convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    tok = model.tokenizer
    ids_list = tok(texts, truncation=False, padding=False, add_special_tokens=True,
                   return_attention_mask=False)["input_ids"]
    lengths = [len(x) for x in ids_list]
    n_overflow = int(sum(l > max_seq_length for l in lengths))

    if long_doc == "exclude":
        keep = [i for i, l in enumerate(lengths) if l <= max_seq_length]
        emb = _enc([texts[i] for i in keep], normalize=True)
        return emb, [doc_ids[i] for i in keep], [family_ids[i] for i in keep], n_overflow

    if long_doc == "mean_pool":
        win = max(1, max_seq_length - 2)   # 특수토큰 여유
        seg_texts, seg_owner = [], []
        for di, ids in enumerate(ids_list):
            if len(ids) <= max_seq_length:
                seg_texts.append(texts[di]); seg_owner.append(di)
            else:
                body = ids[1:-1] if len(ids) >= 2 else ids
                for s in range(0, len(body), win):
                    seg_texts.append(tok.decode(body[s: s + win], skip_special_tokens=True))
                    seg_owner.append(di)
        seg_emb = _enc(seg_texts, normalize=False)   # 평균 전엔 정규화 X
        out = np.zeros((len(texts), seg_emb.shape[1]), np.float32)
        cnt = np.zeros(len(texts))
        for e, owner in zip(seg_emb, seg_owner):
            out[owner] += e; cnt[owner] += 1
        cnt[cnt == 0] = 1
        out /= cnt[:, None]
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)   # 세그먼트 평균 후 L2 정규화
        return out, list(doc_ids), list(family_ids), n_overflow

    # truncate (기본): SBERT 가 max_seq_length 로 head 절단
    return _enc(texts, normalize=True), list(doc_ids), list(family_ids), n_overflow


# 한 prepared-set 을 doc-level 벡터로 인코딩 (set×long_doc 단위 npy 캐시).
# 반환: (emb, meta[doc_id,family_id], timing{encode_sec,source,n_overflow}).
def _encode_set(set_name, docs_df, model_name, max_seq_length, batch_size, device, dtype,
                artifacts_dir, long_doc="truncate", force_rebuild=False):
    cache_root = Path(artifacts_dir) / "cache" / "embeddings_longctx" / model_slug(model_name)
    cache_root.mkdir(parents=True, exist_ok=True)
    # 키에 max_seq_length·dtype 포함 — 설정 바꾸면 새 캐시로 갈려 stale 재사용 방지.
    tag = f"{set_name}__{long_doc}__L{max_seq_length}__{dtype}"
    npy_p = cache_root / f"{tag}.npy"
    meta_p = cache_root / f"{tag}.parquet"
    info_p = cache_root / f"{tag}.meta.json"

    if not force_rebuild and npy_p.exists() and meta_p.exists() and info_p.exists():
        emb = np.load(npy_p)
        meta_df = pd.read_parquet(meta_p)
        info = json.loads(info_p.read_text(encoding="utf-8"))
        return emb, meta_df, {"encode_sec": float(info.get("encode_sec", 0.0)), "source": "cached",
                              "n_overflow": int(info.get("n_overflow", 0))}

    model = _get_encoder(model_name, max_seq_length, device, dtype)
    sub = docs_df[["doc_id", "family_id", "text"]].sort_values("doc_id").reset_index(drop=True)
    texts = [t if str(t).strip() else " " for t in sub["text"].fillna("").astype(str)]

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # 이전 인코딩의 예약메모리 반환 → 파편화로 인한 OOM 완화

    t0 = perf_counter()
    emb, kept_doc, kept_fam, n_overflow = _encode_docs(
        model, texts, sub["doc_id"].tolist(), sub["family_id"].tolist(),
        max_seq_length, batch_size, long_doc)
    encode_sec = perf_counter() - t0

    meta_df = pd.DataFrame({"doc_id": kept_doc, "family_id": kept_fam})
    np.save(npy_p, emb)
    meta_df.to_parquet(meta_p, index=False)
    info_p.write_text(json.dumps({"model_name": model_name, "max_seq_length": int(max_seq_length),
                                  "long_doc": long_doc, "encode_sec": float(encode_sec),
                                  "n_docs": int(len(meta_df)), "n_overflow": int(n_overflow)},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return emb, meta_df, {"encode_sec": float(encode_sec), "source": "live", "n_overflow": int(n_overflow)}


# rival 계약: (reference, query) → (votes_df, timing). 문서 통짜 임베딩을 기밀 풀(FAISS)에 등록 → 벡터서치.
# long_doc: 초과 문서 처리 전략 ("truncate" | "mean_pool" | "exclude").
def longctx_votes(
    reference_docs_df: pd.DataFrame,
    query_docs_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ,
    long_doc: str = "truncate",
    batch_size: int = 32,
    device: str | None = None,
    dtype: str = "float32",
    artifacts_dir: str | Path = "artifacts",
    ref_set_name: str = "reference",
    query_set_name: str = "query",
    faiss_config=None,
    top_k: int = 50,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if long_doc not in LONG_DOC_MODES:
        raise ValueError(f"long_doc must be one of {LONG_DOC_MODES}, got {long_doc!r}")
    if faiss_config is None:
        from sdlp.index.faiss_hnsw import FAISSHNSWConfig
        faiss_config = FAISSHNSWConfig()

    t0 = perf_counter()
    ref_emb, ref_meta, ref_t = _encode_set(ref_set_name, reference_docs_df, model_name, max_seq_length,
                                           batch_size, device, dtype, artifacts_dir, long_doc, force_rebuild)
    q_emb, q_meta, q_t = _encode_set(query_set_name, query_docs_df, model_name, max_seq_length,
                                     batch_size, device, dtype, artifacts_dir, long_doc, force_rebuild)

    # 문서벡터 → 기밀 풀(FAISS) 등록 + 벡터서치 + maxsim (chunk_maxsim 과 동일 tail, docvec 공용).
    from sdlp.methods.docvec import doc_vector_maxsim
    votes, index_build_sec, search_sec = doc_vector_maxsim(
        ref_emb, ref_meta["doc_id"].tolist(), ref_meta["family_id"].tolist(),
        q_emb, q_meta["doc_id"].tolist(), q_meta["family_id"].tolist(),
        faiss_config, top_k=top_k)

    timing = {
        "long_doc": long_doc,
        "ref_encode_sec": ref_t["encode_sec"], "query_encode_sec": q_t["encode_sec"],
        "search_sec": search_sec, "index_build_sec": index_build_sec, "total_sec": perf_counter() - t0,
        # §4: inference = 쿼리 임베딩 + 벡터서치. ref 임베딩·인덱스 빌드는 등록이라 제외.
        "inference_total_sec": q_t["encode_sec"] + search_sec,
        "build_sec": ref_t["encode_sec"] + index_build_sec,
        "ref_source": ref_t["source"], "query_source": q_t["source"],
        "n_overflow_ref": ref_t["n_overflow"], "n_overflow_query": q_t["n_overflow"],
        "model_name": model_name, "max_seq_length": int(max_seq_length),
    }
    return votes, timing
