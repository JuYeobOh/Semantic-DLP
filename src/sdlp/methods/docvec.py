"""문서 단위 임베딩 검색 공용 — 문서벡터를 기밀 풀(FAISS)에 등록하고 벡터서치 + maxsim 집계.

longctx(긴 context 통짜 벡터)·embedding_pooled(50단어 청크 평균 벡터)가 공유한다. 두 기법의 차이는
'문서를 어떻게 1벡터로 만드나'뿐 — 그 뒤 등록·검색·집계는 chunk_maxsim 과 동일한 컴포넌트를 쓴다
(build_faiss_hnsw_index → search_index → aggregate_maxsim_by_document).
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd


# 청크 임베딩(행)들을 doc_id 로 묶어 문서별 평균 + L2 정규화 → (doc_ids, family_ids, doc_mat(N×D)).
def pool_by_doc(chunks_df, emb):
    if len(chunks_df) == 0:
        return [], [], np.zeros((0, emb.shape[1] if emb.ndim == 2 else 0), dtype=np.float32)
    doc_ids = chunks_df["doc_id"].to_numpy()
    fam = chunks_df["family_id"].to_numpy()
    uniq = pd.unique(doc_ids)
    vecs, fams = [], []
    for did in uniq:
        mask = doc_ids == did
        v = emb[mask].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)   # 평균 후 L2 정규화 (내적=코사인 → FAISS inner product)
        vecs.append(v.astype(np.float32))
        fams.append(fam[mask][0])
    return list(uniq), fams, np.stack(vecs)


# 문서벡터를 FAISS 인덱스/검색이 먹는 청크 메타로 (문서 1개 = 청크 1개: chunk_id=doc_id, chunk_index=0).
# dataset 컬럼은 search_index 가 요구만 하고 값은 아무도 안 읽으므로 "" 고정.
def doc_meta(doc_ids, fams):
    return pd.DataFrame({
        "chunk_id": doc_ids, "doc_id": doc_ids, "family_id": fams,
        "dataset": "", "chunk_index": 0,
    })


# ref 문서벡터를 FAISS 풀에 등록(=build) → query 문서벡터 벡터서치(=inference) → maxsim 집계.
# 반환: (votes_df, build_sec, search_sec). 벡터는 L2 정규화 전제(내적=코사인). confidence=최근접 ref 문서 cosine.
def doc_vector_maxsim(ref_emb, ref_doc_ids, ref_fams, q_emb, q_doc_ids, q_fams, faiss_config, top_k=50):
    from sdlp.index.faiss_hnsw import build_faiss_hnsw_index
    from sdlp.retrieval.core import search_index
    from sdlp.voting.core import VOTES_COLUMNS, aggregate_maxsim_by_document

    if len(ref_doc_ids) == 0 or len(q_doc_ids) == 0:
        return pd.DataFrame(columns=VOTES_COLUMNS), 0.0, 0.0

    t_b = perf_counter()
    index = build_faiss_hnsw_index(np.asarray(ref_emb, dtype=np.float32),
                                   doc_meta(ref_doc_ids, ref_fams), faiss_config)
    build_sec = perf_counter() - t_b

    t_s = perf_counter()
    retrieval = search_index(index, doc_meta(q_doc_ids, q_fams),
                             np.asarray(q_emb, dtype=np.float32), min(len(ref_doc_ids), top_k))
    votes = aggregate_maxsim_by_document(retrieval)
    search_sec = perf_counter() - t_s
    return votes, build_sec, search_sec
