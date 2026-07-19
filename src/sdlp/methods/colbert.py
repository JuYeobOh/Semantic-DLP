"""colbertv2 라이벌 (pylate) — 토큰 late-interaction. B2 = 문서 토큰-bag MaxSim mean.

문서를 **토큰 단위**로 모델 최대 길이에 맞춰 청킹(ref≈doc_len 500, query≈query_len 90, 모델 tokenizer)한
뒤 토큰 임베딩을 이어붙여 doc-bag 을 만든다.
- 등록(build): ref 청크 토큰을 **PLAID 인덱스**에 (id=chunk_id, PLAID 는 passage≤512 전제라 청크 단위).
- 후보(Stage1): query 청크 → PLAID retrieve → 그 청크들의 **doc_id 집합** = 후보 ref 문서 (투표 없음).
- 채점(Stage2/B2): query doc-bag ↔ 후보 ref doc-bag **exact MaxSim mean** = confidence.
  Σ_{query 토큰} max_{ref 토큰}(q·d) / query 토큰수 (ColBERT 토큰은 L2정규화 → 내적=cosine).

청킹은 "모델 최대 수용"으로: ref=doc-side([D], ~doc_len=512), query=query-side([Q], ~query_len=96 채움).
query 를 96 까지 채우므로 [MASK] augmentation 은 사실상 없음 → mask 모드 구분 없이 반환 토큰 전부 사용.

confidence=MaxSim mean(raw≥0, best-F1 sweep 로 평가). §4: ref 인코딩+PLAID 빌드=build(제외),
query 인코딩+retrieve+MaxSim=inference. 토큰 임베딩은 set 단위 ragged 디스크 캐시.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sdlp.detection.core import vote_entropy, vote_gini, vote_variance
from sdlp.embedding.spec import model_slug
from sdlp.voting.core import VOTES_COLUMNS

DEFAULT_MODEL = "colbert-ir/colbertv2.0"
QUERY_LEN = 96
DOC_LEN = 512
# 토큰 단위 청킹(모델 tokenizer) — 특수토큰([CLS]/[Q|D]/[SEP]) 여유 두고 모델 최대까지 채움.
REF_TOKENS = 500     # doc-side (< doc_len 512)
QUERY_TOKENS = 90    # query-side (< query_len 96)


# query bag ↔ ref bag MaxSim mean. Q,D: torch [Ntok,128] (L2정규화 → 내적=cosine).
# 긴 문서는 [Qtok,Rtok] 전체 행렬이 폭발(OOM) → query·ref 토큰을 블록으로 나눠 running max (결과 동일).
def _maxsim_mean(Q, D, qblock=4096, dblock=4096) -> float:
    import torch

    nq = int(Q.shape[0])
    if nq == 0 or int(D.shape[0]) == 0:
        return 0.0
    with torch.no_grad():
        total = 0.0
        for qi in range(0, nq, qblock):
            Qb = Q[qi:qi + qblock]
            run = None   # 이 query 블록의 각 토큰이 본 ref 최대 내적
            for di in range(0, int(D.shape[0]), dblock):
                m = (Qb @ D[di:di + dblock].T).max(dim=1).values   # [qb]
                run = m if run is None else torch.maximum(run, m)
            total += float(run.sum().item())
        return total / nq


# retrieve 결과(쿼리 청크별 top-k ref 청크) → 쿼리 문서별 후보 ref 문서 상위 shortlist.
# 순위 = 그 ref 문서를 가리킨 query 청크 hit **개수**(참조본 value_counts 방식). 여러 청크가 겹칠수록
# 문서 전체 정합성이 높다는 신호. 후보 선택만 count 이고, 최종 confidence 는 여전히 exact MaxSim(투표 아님).
def _candidate_docs(results, q_chunk_docs, ref_chunk_to_doc, shortlist_k):
    per_qdoc: dict[object, dict] = {}   # query_doc -> {ref_doc: hit_count}
    for i, res in enumerate(results):
        d = per_qdoc.setdefault(q_chunk_docs[i], {})
        for r in res:
            rcid = r["id"] if isinstance(r, dict) else r.id
            rdoc = ref_chunk_to_doc.get(rcid)
            if rdoc is None:
                continue
            d[rdoc] = d.get(rdoc, 0) + 1
    # count 내림차순, 동점은 doc_id 로 결정적.
    return {qd: [rd for rd, _ in sorted(dd.items(), key=lambda x: (-x[1], str(x[0])))[:shortlist_k]]
            for qd, dd in per_qdoc.items()}


# 한 prepared **set 전체**를 청크 토큰 임베딩으로 (set 단위 ragged 캐시 — split·변형·inclorig 무관).
# is_query 로 청킹 길이(query_len/doc_len)와 [Q]/[D] 인코딩이 갈리므로 캐시도 side 별로 분리.
# 반환: (meta[chunk_id,doc_id,family_id,start,end], embs 리스트). 미스 시에만 set 로드+인코딩.
def _encode_set(get_model, set_name, is_query, artifacts_dir, prepared_dir, model_name, batch_size):
    from sdlp.chunking.core import ChunkSpec, build_chunks_df

    n_tok = QUERY_TOKENS if is_query else REF_TOKENS
    side = "q" if is_query else "d"
    slug = f"tok{n_tok}o0__{side}"
    cache_dir = Path(artifacts_dir) / "cache" / "embeddings_colbert" / model_slug(model_name) / slug / set_name
    meta_p, tok_p = cache_dir / "meta.parquet", cache_dir / "tokens.npy"
    if meta_p.exists() and tok_p.exists():
        meta = pd.read_parquet(meta_p)
        flat = np.load(tok_p)
        embs = [flat[r.start:r.end] for r in meta.itertuples(index=False)]
        return meta, embs

    from sdlp.io import load_prepared_set
    docs_df = load_prepared_set(prepared_dir, set_name)
    model = get_model()   # 캐시 미스일 때만 모델 로드 (토큰 청킹에 tokenizer 필요)
    chunks_df = build_chunks_df(docs_df, ChunkSpec(mode="token", size=n_tok, overlap=0),
                                tokenizer=model.tokenizer).reset_index(drop=True)
    texts = chunks_df["chunk_text"].astype(str).tolist()
    raw = model.encode(texts, is_query=is_query, convert_to_tensor=True,
                       batch_size=batch_size, show_progress_bar=True)
    embs = ([raw[i].cpu().numpy().astype(np.float16) for i in range(len(texts))]
            if not isinstance(raw, list) else [e.cpu().numpy().astype(np.float16) for e in raw])

    starts, cur = [], 0
    for e in embs:
        starts.append(cur); cur += e.shape[0]
    meta = chunks_df[["chunk_id", "doc_id", "family_id"]].copy()
    meta["start"] = starts
    meta["end"] = [s + e.shape[0] for s, e in zip(starts, embs)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(tok_p, np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), np.float16))
    meta.to_parquet(meta_p, index=False)
    return meta, embs


# 여러 set 을 인코딩(캐시)해 doc_id → 청크 임베딩 매핑으로 합침. 반환: (embs_by_doc, fam_by_doc).
# embs_by_doc[doc_id] = 그 문서 청크 임베딩 리스트(순서 보존).
def _encode_sets_by_doc(get_model, set_names, is_query, artifacts_dir, prepared_dir, model_name, batch_size):
    embs_by_doc: dict[object, list] = {}
    fam_by_doc: dict[object, object] = {}
    for s in set_names:
        meta, embs = _encode_set(get_model, s, is_query, artifacts_dir, prepared_dir, model_name, batch_size)
        for i, r in enumerate(meta.itertuples(index=False)):
            embs_by_doc.setdefault(r.doc_id, []).append((r.chunk_id, embs[i]))
            fam_by_doc[r.doc_id] = r.family_id
    return embs_by_doc, fam_by_doc


# retrieve 를 위해 doc_id 별 청크 임베딩을 flat 리스트로 (documents_ids=chunk_id, doc 매핑 동반).
def _flatten_chunks(embs_by_doc):
    chunk_ids, chunk_embs, chunk_to_doc = [], [], {}
    for doc_id, items in embs_by_doc.items():
        for cid, emb in items:
            chunk_ids.append(cid); chunk_embs.append(emb); chunk_to_doc[cid] = doc_id
    return chunk_ids, chunk_embs, chunk_to_doc


# rival 계약: (reference, query) → (votes_df, timing). ColBERT B2 (문서 bag MaxSim mean).
# set_names: 이 run 을 구성하는 prepared set 들(원본+변형). set 단위로 인코딩(캐시) 후 ref/query doc_id 로 조립.
def colbert_votes(reference_docs_df, query_docs_df, set_names, prepared_dir="data/prepared",
                  model_name=DEFAULT_MODEL, shortlist_k=32, retrieve_k=1, artifacts_dir="artifacts",
                  batch_size=32):
    import torch
    from pylate import indexes, retrieve

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델은 캐시 미스로 실제 인코딩할 때만 로드(1회) — 전 set 캐시면 모델·GPU 안 씀. PLAID/retrieve 는 모델 불필요.
    _box: dict = {}

    def get_model():
        if "m" not in _box:
            from pylate import models
            _box["m"] = models.ColBERT(model_name_or_path=model_name,
                                       query_length=QUERY_LEN, document_length=DOC_LEN)
        return _box["m"]

    ref_ids = set(reference_docs_df["doc_id"])
    q_ids = set(query_docs_df["doc_id"])
    q_fam_of = dict(zip(query_docs_df["doc_id"], query_docs_df["family_id"]))

    # 문서 토큰 bag 조립 (lazy, GPU) — 반환 토큰 전부 사용(마스크 모드 없음).
    def _bag(items):
        return torch.from_numpy(np.concatenate([e for _, e in items], axis=0)).float().to(device)

    # ---- ref: set 단위 인코딩(캐시) → 청크를 PLAID 등록 (=build, 추론 latency 제외) ----
    t0 = perf_counter()
    ref_by_doc, ref_fam = _encode_sets_by_doc(get_model, set_names, False, artifacts_dir, prepared_dir,
                                              model_name, batch_size)
    ref_by_doc = {d: v for d, v in ref_by_doc.items() if d in ref_ids}     # 기밀 절반만
    ref_cids, ref_cembs, ref_chunk_to_doc = _flatten_chunks(ref_by_doc)
    index = indexes.PLAID(index_folder=str(Path(artifacts_dir) / "colbert_index"),
                          index_name="colbert_ref", override=True, show_progress=False)
    index.add_documents(documents_ids=ref_cids,
                        documents_embeddings=[e.astype(np.float32) for e in ref_cembs])
    build_sec = perf_counter() - t0

    # ---- query: set 단위 인코딩(캐시) → retrieve(후보) → MaxSim (=inference) ----
    t1 = perf_counter()
    q_by_doc, _ = _encode_sets_by_doc(get_model, set_names, True, artifacts_dir, prepared_dir,
                                      model_name, batch_size)
    q_by_doc = {d: v for d, v in q_by_doc.items() if d in q_ids}           # positive+benign 만
    q_cids, q_cembs, q_chunk_to_qdoc = _flatten_chunks(q_by_doc)
    retriever = retrieve.ColBERT(index=index)
    results = retriever.retrieve(queries_embeddings=[e.astype(np.float32) for e in q_cembs], k=retrieve_k)
    cand_by_qdoc = _candidate_docs(results, [q_chunk_to_qdoc[c] for c in q_cids],
                                   ref_chunk_to_doc, shortlist_k)

    rows = []
    for qd, q_items in q_by_doc.items():
        Q = _bag(q_items)
        family_scores: dict[object, float] = {}
        family_best_doc: dict[object, object] = {}
        for rd in cand_by_qdoc.get(qd, []):
            if rd not in ref_by_doc:
                continue
            s = _maxsim_mean(Q, _bag(ref_by_doc[rd]))
            fid = ref_fam.get(rd)
            if fid not in family_scores or s > family_scores[fid]:
                family_scores[fid] = s
                family_best_doc[fid] = rd
        del Q
        if family_scores:
            pred_fam = max(family_scores, key=lambda f: (family_scores[f], str(f)))
            best = family_scores[pred_fam]
            pred_doc = family_best_doc[pred_fam]
            matched = 1
        else:
            pred_fam, pred_doc, best, matched = None, None, 0.0, 0
        dist = np.array(list(family_scores.values()), dtype=float)
        rows.append({
            "query_doc_id": qd, "query_family_id": q_fam_of.get(qd, ""),
            "pred_doc_id": pred_doc, "pred_family_id": pred_fam,
            "n_chunks": len(q_items), "n_votes": matched, "best_votes": matched,
            "confidence": float(best),   # MaxSim mean (best-F1 sweep 로 평가)
            "vote_entropy": vote_entropy(dist), "vote_variance": vote_variance(dist), "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(
                {str(k2): round(float(v), 4) for k2, v in family_scores.items()}, ensure_ascii=False),
        })
    inference_sec = perf_counter() - t1
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec, "shortlist_k": shortlist_k}
    return pd.DataFrame(rows, columns=VOTES_COLUMNS), timing
