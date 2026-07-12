"""MinHash + LSH 라이벌 — 문서 word-shingle 집합의 Jaccard 를 MinHash 로 추정, LSH 로 후보 생성.

각 문서를 k-word shingle 집합으로 보고 MinHash 서명(num_perm)을 만든다. 기밀 ref 를 **MinHashLSH 인덱스에
등록**(=풀), query 는 LSH 로 **후보 ref 만 추린 뒤**(O(N²) 아님) Jaccard 추정치 최대인 family 선택.
confidence = best-family Jaccard 추정(0~1, best-F1 sweep 로 평가). near-duplicate 탐지형 baseline
(verbatim·경미한 편집엔 강하고 심한 패러프레이즈엔 약함 — 임베딩 방법과 대비).

§4: ref MinHash+LSH 등록=build(제외), query MinHash+LSH 질의+Jaccard=inference.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sdlp.detection.core import vote_entropy, vote_gini, vote_variance
from sdlp.voting.core import VOTES_COLUMNS


# 텍스트 → k-word shingle 집합. 빈 텍스트는 빈 집합.
# <k 단어면 전체를 한 shingle 로 (원소 포맷을 k-shingle 과 통일 — unigram 폴백은 긴 문서와 교집합 0).
def _shingles(text: str, k: int) -> set[str]:
    toks = text.split()
    if len(toks) >= k:
        return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}
    return {" ".join(toks)} if toks else set()


# 텍스트 → MinHash 서명 (빈 집합이면 None).
def _minhash(text: str, num_perm: int, k: int):
    from datasketch import MinHash

    sh = _shingles(text, k)
    if not sh:
        return None
    m = MinHash(num_perm=num_perm)
    for s in sh:
        m.update(s.encode("utf8"))
    return m


# 기밀 ref 를 MinHashLSH 에 등록 → query 후보 생성 + Jaccard 추정 → family max-score (VOTES_COLUMNS 반환).
def minhash_lsh_votes(reference_docs_df, query_docs_df, num_perm=128, threshold=0.2, shingle_k=5):
    from datasketch import MinHashLSH

    ref_doc = reference_docs_df["doc_id"].to_numpy()
    ref_fam = reference_docs_df["family_id"].to_numpy()
    ref_texts = [str(t or "").strip() for t in reference_docs_df["text"].fillna("")]

    # ---- ref MinHash + LSH 등록 (=build) ----
    t0 = perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ref_mh: dict[str, object] = {}
    fam_of: dict[str, str] = {}
    for i, text in enumerate(ref_texts):
        mh = _minhash(text, num_perm, shingle_k)
        if mh is None:
            continue
        did = ref_doc[i]
        lsh.insert(did, mh)
        ref_mh[did] = mh
        fam_of[did] = ref_fam[i]
    build_sec = perf_counter() - t0

    # ---- query MinHash + LSH 질의 + Jaccard (=inference) ----
    q_ids = query_docs_df["doc_id"].to_numpy()
    q_fams = query_docs_df["family_id"].to_numpy()
    q_texts = [str(t or "").strip() for t in query_docs_df["text"].fillna("")]
    t1 = perf_counter()
    rows = []
    for qi in tqdm(range(len(query_docs_df)), desc="minhash+lsh query", leave=False):
        family_scores: dict[str, float] = {}
        family_best_doc: dict[str, object] = {}   # family 최고점 문서 → pred_doc 을 pred_fam 안에서 뽑기 위함
        mh = _minhash(q_texts[qi], num_perm, shingle_k)
        if mh is not None:
            for cand in lsh.query(mh):          # LSH 후보 ref (O(N²) 회피)
                score = float(mh.jaccard(ref_mh[cand]))   # Jaccard 추정
                fid = fam_of[cand]
                if fid not in family_scores or score > family_scores[fid]:
                    family_scores[fid] = score
                    family_best_doc[fid] = cand
        if family_scores:
            # 동점 시 family_id 로 결정적 선택 (MinHash 는 1/num_perm 양자화라 동점 흔함).
            pred_fam = max(family_scores, key=lambda f: (family_scores[f], str(f)))
            best = family_scores[pred_fam]
            pred_doc = family_best_doc[pred_fam]   # pred_doc 은 반드시 pred_fam 소속
            matched = 1
        else:
            pred_fam, pred_doc, best, matched = None, None, 0.0, 0
        dist = np.array(list(family_scores.values()), dtype=float)
        rows.append({
            "query_doc_id": q_ids[qi], "query_family_id": q_fams[qi],
            "pred_doc_id": pred_doc, "pred_family_id": pred_fam,
            "n_chunks": 1, "n_votes": matched, "best_votes": matched,   # 후보 없으면 0 (maximum-similarity 라 vote 최소)
            "confidence": float(best),   # best-family Jaccard 추정 (best-F1 sweep 로 평가)
            "vote_entropy": vote_entropy(dist), "vote_variance": vote_variance(dist), "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(
                {str(k2): round(float(v), 4) for k2, v in family_scores.items()}, ensure_ascii=False),
        })
    inference_sec = perf_counter() - t1
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec,
              "num_perm": num_perm, "threshold": threshold, "shingle_k": shingle_k}
    return pd.DataFrame(rows, columns=VOTES_COLUMNS), timing
