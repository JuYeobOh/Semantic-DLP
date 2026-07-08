"""문서별 투표 집계 — 검색 결과를 쿼리 문서 단위로 모아 최다 득표 문서 + confidence.

- 쿼리 청크의 top-k 이웃 각각이 자기가 가리키는 ref 문서에 한 표. K(청크당 이웃 수)는
  검색(search_index top_k)에서 정해지고, 투표는 retrieval_df 의 모든 행을 집계 → K 를 키우면
  분포가 촘촘해진다.
- confidence = 최다 득표 문서의 표 비율 = max_i p_i(q). (K=1 이면 논문 정의 "최다 득표수/전체 청크수"와 동일)
- use_score_weight=True 면 표 개수 대신 유사도 점수 합으로 집계 (score-weighted voting, 껐다 켤 수 있음).
"""
from __future__ import annotations

import json

import pandas as pd

from sdlp.detection.core import vote_entropy, vote_gini, vote_variance

# 투표 집계 결과 스키마 (빈 입력일 때도 동일 컬럼).
VOTES_COLUMNS = [
    "query_doc_id", "query_family_id", "pred_doc_id", "pred_family_id",
    "n_chunks", "n_votes", "best_votes", "confidence",
    "vote_entropy", "vote_variance", "vote_gini", "vote_distribution_json",
]


# 검색 long-form 을 쿼리 문서 단위로 집계 → 최다 득표 문서 + confidence + 분포 보조통계.
# use_score_weight: False=표 개수, True=유사도 점수 합.
def vote_by_document(retrieval_df: pd.DataFrame, use_score_weight: bool = False) -> pd.DataFrame:
    if len(retrieval_df) == 0:
        return pd.DataFrame(columns=VOTES_COLUMNS)

    rows: list[dict] = []
    for query_doc_id, g in retrieval_df.groupby("query_doc_id", sort=False):
        # 문서별 집계값 (내림차순): score-weight 면 점수 합, 아니면 표 개수.
        if use_score_weight:
            agg = g.groupby("ref_doc_id")["score"].sum().sort_values(ascending=False)
        else:
            agg = g["ref_doc_id"].value_counts()

        pred_doc_id = agg.index[0]
        total = float(agg.sum())
        confidence = float(agg.iloc[0]) / total if total > 0 else 0.0
        best_votes = int((g["ref_doc_id"] == pred_doc_id).sum())
        pred_family_id = g.loc[g["ref_doc_id"] == pred_doc_id, "ref_family_id"].iloc[0]
        dist = agg.to_numpy()

        rows.append({
            "query_doc_id": query_doc_id,
            "query_family_id": g["query_family_id"].iloc[0],
            "pred_doc_id": pred_doc_id,
            "pred_family_id": pred_family_id,
            "n_chunks": int(g["query_chunk_id"].nunique()),   # 쿼리 청크 수 Kq
            "n_votes": int(len(g)),                            # 집계된 총 표 수 (= Kq × K 근사)
            "best_votes": best_votes,
            "confidence": confidence,
            "vote_entropy": vote_entropy(dist),
            "vote_variance": vote_variance(dist),
            "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(agg.to_dict(), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


# chunk_maxsim 집계 — chunk_voting 과 검색은 동일, 집계만 다름(투표수 대신 최대 유사도).
# confidence = 그 쿼리 문서의 모든 (청크, 이웃) 쌍 중 최대 cosine. pred = 그 최대 쌍의 ref 문서/family.
# vote_distribution_json = ref 문서별 최대 유사도(내림차순). 보조통계는 그 분포의 집중도.
def aggregate_maxsim_by_document(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if len(retrieval_df) == 0:
        return pd.DataFrame(columns=VOTES_COLUMNS)

    rows: list[dict] = []
    for query_doc_id, g in retrieval_df.groupby("query_doc_id", sort=False):
        # 문서별 최대 유사도 (내림차순). confidence 는 전체 최대(=1등 문서의 최대).
        agg = g.groupby("ref_doc_id")["score"].max().sort_values(ascending=False)
        pred_doc_id = agg.index[0]
        confidence = float(agg.iloc[0])
        pred_family_id = g.loc[g["ref_doc_id"] == pred_doc_id, "ref_family_id"].iloc[0]
        best_votes = int((g["ref_doc_id"] == pred_doc_id).sum())   # 참고용(득표 아님)
        dist = agg.to_numpy()

        rows.append({
            "query_doc_id": query_doc_id,
            "query_family_id": g["query_family_id"].iloc[0],
            "pred_doc_id": pred_doc_id,
            "pred_family_id": pred_family_id,
            "n_chunks": int(g["query_chunk_id"].nunique()),
            "n_votes": int(len(g)),                            # 집계된 (청크,이웃) 쌍 수
            "best_votes": best_votes,
            "confidence": confidence,
            "vote_entropy": vote_entropy(dist),
            "vote_variance": vote_variance(dist),
            "vote_gini": vote_gini(dist),
            "vote_distribution_json": json.dumps(
                {k: round(float(v), 6) for k, v in agg.to_dict().items()}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)
