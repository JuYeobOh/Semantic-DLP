"""BM25 라이벌 (bm25s, sparse) — full-doc + keyword(기밀 gold KP 인덱스) 두 scope.

- scope="doc": **인덱스 = 기밀 ref 전문**, 쿼리 = query 전문.
- scope="keyword": **인덱스 = 기밀 ref 의 gold keyphrases**(핵심어 지문), 쿼리 = query **전문**.
  → "유출(변형) 텍스트가 어느 기밀 문서의 핵심어를 담고 있나" 를 측정. 양측 Snowball stemmer
  (형태 변형 매칭). gold KP 는 원본 set meta_json["keywords"](';' 구분).

confidence = raw best-family BM25 score (best-F1 sweep 로 평가). §4: index=build(제외), 검색=inference.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd

from sdlp.detection.core import vote_entropy, vote_gini, vote_variance
from sdlp.voting.core import VOTES_COLUMNS


# family_id -> gold keyphrase 문자열 (원본 set meta_json["keywords"], ';' → 공백). keyword scope 전용.
# keyword 없는 데이터셋(re3/par3/casimir)은 ValueError → 표에서 N.A.
def load_keyphrases_by_family(prepared_dir, original_set: str) -> dict[str, str]:
    from sdlp.io import load_prepared_set

    df = load_prepared_set(prepared_dir, original_set)
    mapping, n_kw = {}, 0
    for row in df.itertuples(index=False):
        meta = json.loads(row.meta_json) if isinstance(row.meta_json, str) else (row.meta_json or {})
        kw = meta.get("keywords")
        if kw:
            n_kw += 1
        mapping[row.family_id] = " ".join(str(kw or "").split(";"))
    if n_kw == 0:
        raise ValueError(f"{original_set} 에 gold keyphrases 없음 — keyword BM25 미지원(keyphrase 데이터셋만)")
    return mapping


# Snowball(english) stemmer (keyword scope 에서 ref·query 양측 동일 적용).
def _stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


# ref 전문 BM25 인덱스 → query 검색 → family max-score 집계 (VOTES_COLUMNS 반환).
def bm25_votes(reference_docs_df, query_docs_df, scope="doc",
               keyphrase_by_family=None, k1=1.5, b=0.75, top_k=200):
    import bm25s

    stemmer = _stemmer() if scope == "keyword" else None
    ref_doc = reference_docs_df["doc_id"].to_numpy()
    ref_fam = reference_docs_df["family_id"].to_numpy()

    # 인덱스 코퍼스: doc=기밀 ref 전문 / keyword=기밀 ref 의 gold KP(핵심어 지문). 쿼리는 항상 query 전문.
    if scope == "keyword":
        if keyphrase_by_family is None:
            raise ValueError("keyword scope 는 keyphrase_by_family 필요")
        corpus_texts = [keyphrase_by_family.get(f, "") for f in ref_fam]
    else:
        corpus_texts = [str(t or "").strip() for t in reference_docs_df["text"].fillna("")]
    query_texts = [str(t or "").strip() for t in query_docs_df["text"].fillna("")]
    if not corpus_texts:
        return pd.DataFrame(columns=VOTES_COLUMNS), {"inference_total_sec": 0.0, "build_sec": 0.0}

    # ---- 인덱스(=등록, build) ----
    t0 = perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=False)
    build_sec = perf_counter() - t0

    # ---- query 토큰화 + 검색(=inference) ----
    t1 = perf_counter()
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    k = min(len(corpus_texts), top_k)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    inference_sec = perf_counter() - t1

    # ---- family max-score 집계 ----
    q_ids = query_docs_df["doc_id"].to_numpy()
    q_fams = query_docs_df["family_id"].to_numpy()
    rows = []
    for qi in range(len(query_docs_df)):
        family_scores: dict[str, float] = {}
        doc_scores: dict[str, float] = {}
        for idx, s in zip(results[qi], scores[qi]):
            s = float(s)
            if s <= 0:
                continue
            fid, did = ref_fam[idx], ref_doc[idx]
            if fid not in family_scores or s > family_scores[fid]:
                family_scores[fid] = s
            if did not in doc_scores or s > doc_scores[did]:
                doc_scores[did] = s
        if family_scores:
            pred_fam = max(family_scores, key=family_scores.get)
            best = family_scores[pred_fam]
            pred_doc = max(doc_scores, key=doc_scores.get)
        else:
            pred_fam, pred_doc, best = None, None, 0.0
        dist = np.array(list(family_scores.values()), dtype=float)
        rows.append({
            "query_doc_id": q_ids[qi], "query_family_id": q_fams[qi],
            "pred_doc_id": pred_doc, "pred_family_id": pred_fam,
            "n_chunks": 1, "n_votes": 1, "best_votes": 1,
            "confidence": float(best),   # raw BM25 (best-F1 sweep 로 평가)
            "vote_entropy": vote_entropy(dist), "vote_variance": vote_variance(dist), "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(
                {k2: round(float(v), 4) for k2, v in family_scores.items()}, ensure_ascii=False),
        })
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec,
              "index_sec": build_sec, "search_sec": inference_sec, "scope": scope}
    return pd.DataFrame(rows), timing
