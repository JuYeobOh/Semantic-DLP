"""벡터 검색 — 쿼리 청크별 top-k 최근접 기밀(ref) 청크를 long-form DataFrame 으로.

각 (쿼리 청크, ref 청크) 페어가 한 행. 이후 투표(S5)가 이 표를 문서 단위로 집계한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 검색 결과 long-form 스키마 (빈 결과일 때도 동일 컬럼 유지).
RETRIEVAL_COLUMNS = [
    "query_chunk_id", "query_doc_id", "query_family_id", "query_chunk_index",
    "ref_chunk_id", "ref_doc_id", "ref_family_id", "ref_dataset", "ref_chunk_index",
    "score", "rank",
]


# 쿼리 청크 임베딩을 인덱스에서 top_k 검색 → (쿼리청크, ref청크) 페어 long-form.
def search_index(
    index_artifact,
    query_chunks_df: pd.DataFrame,
    query_embeddings: np.ndarray,
    top_k: int = 1,
) -> pd.DataFrame:
    if len(query_chunks_df) != len(query_embeddings):
        raise ValueError("query_chunks_df 와 query_embeddings 길이가 일치해야 함")
    if len(query_chunks_df) == 0:
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)

    scores, indices = index_artifact.search(query_embeddings.astype(np.float32), top_k)
    ref = index_artifact.meta_df.reset_index(drop=True)
    ref_chunk_ids = ref["chunk_id"].to_numpy()
    ref_doc_ids = ref["doc_id"].to_numpy()
    ref_family_ids = ref["family_id"].to_numpy()
    ref_datasets = ref["dataset"].to_numpy()
    ref_chunk_indices = ref["chunk_index"].to_numpy()

    rows: list[dict] = []
    for q_idx, qrow in enumerate(query_chunks_df.itertuples(index=False)):
        for rank, (idx, score) in enumerate(zip(indices[q_idx], scores[q_idx]), start=1):
            if idx < 0:  # 결과 부족 시 faiss 는 -1 반환 → 건너뜀
                continue
            rows.append({
                "query_chunk_id": qrow.chunk_id,
                "query_doc_id": qrow.doc_id,
                "query_family_id": qrow.family_id,
                "query_chunk_index": qrow.chunk_index,
                "ref_chunk_id": ref_chunk_ids[idx],
                "ref_doc_id": ref_doc_ids[idx],
                "ref_family_id": ref_family_ids[idx],
                "ref_dataset": ref_datasets[idx],
                "ref_chunk_index": int(ref_chunk_indices[idx]),
                "score": float(score),
                "rank": rank,
            })
    return pd.DataFrame(rows)
