"""ssdeep(ppdeep) 퍼지 해시 라이벌 — 문서 전문 fuzzy hash → query×ref 최대 유사도 매칭.

각 문서를 ppdeep 컨텍스트 트리거 조각 해시로 만들고, query 해시를 모든 ref 해시와 비교(0~100).
query 별 family 최대 점수 → pred_family, confidence = best/100. 임베딩·벡터인덱스 없음(해시 O(N²) 비교).
confidence 는 best-F1 sweep 로 평가. §4: ref 해싱=build(등록, 제외), query 해싱+비교=inference.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sdlp.detection.core import vote_entropy, vote_gini, vote_variance
from sdlp.voting.core import VOTES_COLUMNS


# 텍스트 리스트 → ppdeep 해시 리스트 (빈 텍스트는 None).
def _hash_texts(texts):
    import ppdeep

    return [ppdeep.hash(t) if t else None for t in texts]


# ref 전문 해시 → query 해시와 전량 비교 → family max-score 집계 (VOTES_COLUMNS 반환).
def ssdeep_votes(reference_docs_df, query_docs_df):
    import ppdeep

    ref_doc = reference_docs_df["doc_id"].to_numpy()
    ref_fam = reference_docs_df["family_id"].to_numpy()
    ref_texts = [str(t or "").strip() for t in reference_docs_df["text"].fillna("")]

    # ---- ref 해싱 (=등록/build) ----
    t0 = perf_counter()
    ref_hashes = _hash_texts(ref_texts)
    ref_entries = [(h, ref_doc[i], ref_fam[i]) for i, h in enumerate(ref_hashes) if h]
    build_sec = perf_counter() - t0

    # ---- query 해싱 + 비교 (=inference) ----
    q_ids = query_docs_df["doc_id"].to_numpy()
    q_fams = query_docs_df["family_id"].to_numpy()
    q_texts = [str(t or "").strip() for t in query_docs_df["text"].fillna("")]
    t1 = perf_counter()
    q_hashes = _hash_texts(q_texts)
    rows = []
    # ponytail: single-thread O(N_ref × N_query) 비교. 대규모(casimir)에서 느리면 blocksize 버킷팅
    # (호환 blocksize끼리만 비교 → 결과 동일) 또는 query 단위 ProcessPoolExecutor 로 병렬화.
    for qi in tqdm(range(len(query_docs_df)), desc="ssdeep compare", leave=False):
        qh = q_hashes[qi]
        family_scores: dict[str, int] = {}
        doc_scores: dict[str, int] = {}
        if qh:
            for r_hash, r_doc, r_fam in ref_entries:
                score = ppdeep.compare(qh, r_hash)
                if score <= 0:
                    continue
                if r_fam not in family_scores or score > family_scores[r_fam]:
                    family_scores[r_fam] = score
                if r_doc not in doc_scores or score > doc_scores[r_doc]:
                    doc_scores[r_doc] = score
        if family_scores:
            pred_fam = max(family_scores, key=family_scores.get)
            best = family_scores[pred_fam]
            pred_doc = max(doc_scores, key=doc_scores.get)
        else:
            pred_fam, pred_doc, best = None, None, 0
        dist = np.array(list(family_scores.values()), dtype=float)
        rows.append({
            "query_doc_id": q_ids[qi], "query_family_id": q_fams[qi],
            "pred_doc_id": pred_doc, "pred_family_id": pred_fam,
            "n_chunks": 1, "n_votes": 1, "best_votes": 1,
            "confidence": float(best) / 100.0,   # ppdeep 0~100 → 0~1 (best-F1 sweep 로 평가)
            "vote_entropy": vote_entropy(dist), "vote_variance": vote_variance(dist), "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(
                {k: int(v) for k, v in family_scores.items()}, ensure_ascii=False),
        })
    inference_sec = perf_counter() - t1
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec}
    return pd.DataFrame(rows, columns=VOTES_COLUMNS), timing
