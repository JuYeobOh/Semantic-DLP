"""embedding_pooled — 50단어 청크를 문서별 mean-pool 한 문서벡터를 기밀 풀(FAISS)에 등록 → 벡터서치.

chunk_maxsim 의 **문서 단위 버전**. 문서벡터: chunk_voting 과 같은 50단어 청크를 raw(normalize=False)
임베딩 → 평균 → L2정규화 (longctx mean_pool 과 같은 A방식). raw 벡터는 cache/embeddings_raw/ 로 갈려
기존 정규화 캐시와 안 섞인다. 등록·검색·집계(=chunk_maxsim 동일)는 docvec.doc_vector_maxsim 공용.

§4 latency: 인덱스 빌드=등록(build, 제외), 쿼리 임베딩+벡터서치=inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ref 문서벡터를 기밀 풀(FAISS)에 등록 → query 문서벡터 벡터서치 → maxsim 집계. cfg 로 청크 캐시 재사용.
def embedding_pooled_votes(reference_df, query_df, cfg, top_k=50):
    from dataclasses import replace

    from sdlp.embedding.st import STTextEmbedder
    from sdlp.methods.docvec import doc_vector_maxsim, pool_by_doc
    from sdlp.pipeline.run import _filter, _get_or_embed

    # raw(normalize=False) 임베딩을 별도 캐시(embeddings_raw/)에서 — longctx mean_pool 과 같은 A방식.
    cfg = replace(cfg, embed_spec=replace(cfg.embed_spec, normalize_embeddings=False))
    embedder = STTextEmbedder(cfg.embed_spec)   # 메인 파이프라인과 동일하게 항상 생성
    ref_ids = set(reference_df["doc_id"])
    q_ids = set(query_df["doc_id"])

    # ---- ref 청크(원본 set) → 문서벡터 ----
    orig_chunks, orig_emb, _c, embed_ref_sec, src_o = _get_or_embed(cfg, cfg.resolved_original_set, embedder)
    dim = orig_emb.shape[1] if orig_emb.ndim == 2 else 0
    ref_c, ref_e = _filter(orig_chunks, orig_emb, ref_ids)
    ref_doc_ids, ref_fam, ref_mat = pool_by_doc(ref_c, ref_e)

    # ---- query 청크(원본 query + 변형 set) → 문서벡터. 임베딩 시간은 캐시 기록값 ----
    parts_c, parts_e, embed_query_sec, sources = [], [], 0.0, [src_o]
    oc, oe = _filter(orig_chunks, orig_emb, q_ids)
    if len(oc):
        parts_c.append(oc); parts_e.append(oe)
    for vset in cfg.resolved_variant_sets:
        v_chunks, v_emb, _c, e_s, src = _get_or_embed(cfg, vset, embedder)
        embed_query_sec += e_s; sources.append(src)
        vc, ve = _filter(v_chunks, v_emb, q_ids)
        if len(vc):
            parts_c.append(vc); parts_e.append(ve)
    query_chunks = pd.concat(parts_c, ignore_index=True) if parts_c else ref_c.iloc[0:0]
    query_emb = np.concatenate(parts_e, axis=0) if parts_e else np.zeros((0, dim), dtype=np.float32)
    q_doc_ids, q_fam, q_mat = pool_by_doc(query_chunks, query_emb)

    # ---- 문서벡터 → FAISS 풀 등록 + 벡터서치 + maxsim (공용, chunk_maxsim 과 동일 tail) ----
    votes, build_sec, search_sec = doc_vector_maxsim(
        ref_mat, ref_doc_ids, ref_fam, q_mat, q_doc_ids, q_fam, cfg.faiss_config, top_k=top_k)

    timing = {
        "inference_total_sec": embed_query_sec + search_sec,   # §4: 쿼리 임베딩 + 벡터서치
        "build_sec": build_sec,
        "embed_ref_sec": float(embed_ref_sec),
        "embed_query_sec": float(embed_query_sec),
        "search_sec": float(search_sec),
        "embed_source": "cached" if set(sources) == {"cached"} else ("live" if set(sources) == {"live"} else "mixed"),
    }
    return votes, timing
