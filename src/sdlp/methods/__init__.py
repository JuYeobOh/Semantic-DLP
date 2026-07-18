"""대안 method 패키지 — chunk_voting 과 동일한 votes_df 를 만드는 drop-in 구현들.

성격 라벨(제안/baseline) 없이 이름만. 분류는 리포트 단계 category 로만 정한다.
각 method: xxx_votes(reference_docs_df, query_docs_df, ...) -> (votes_df, timing).
votes_df 는 sdlp.voting.core.VOTES_COLUMNS 호환(confidence 컬럼) → 같은 evaluate_run 으로 평가.
run_method 로 감싸면 경량 run_dir(votes/manifest/per_query_eval/metrics.json)에 저장 →
chunk_voting 과 동일한 latency·F1·confusion·케이스 조회 셀 재사용.
"""
from sdlp.methods.longctx import DEFAULT_MAX_SEQ, DEFAULT_MODEL, longctx_votes
from sdlp.methods.runner import method_run_dir, run_method, save_method_run


# longctx builder — cfg.method_params(model/long_doc/max_seq_length/dtype/batch_size) → (method_fn, run_tag).
def _build_longctx(cfg):
    p = cfg.method_params
    model = p.get("model", DEFAULT_MODEL)
    mode = p.get("long_doc", "truncate")
    max_seq = int(p.get("max_seq_length", DEFAULT_MAX_SEQ))
    dtype = p.get("dtype", "bfloat16")   # 캐시·기존 run 이 전부 bfloat16 → 기본값 일치(안 넘겨도 hit)
    batch = int(p.get("batch_size", 8))
    # 캐시는 prepared set 단위(키=set명) — 변형별 run 이 같은 set 캐시를 공유, 섞임 원천 차단.
    set_names = [cfg.resolved_original_set, *cfg.resolved_variant_sets]

    def method_fn(reference_df, query_df):
        return longctx_votes(
            reference_df, query_df, set_names=set_names, prepared_dir=cfg.prepared_dir,
            long_doc=mode, model_name=model, max_seq_length=max_seq,
            dtype=dtype, batch_size=batch, artifacts_dir=cfg.artifacts_dir,
            faiss_config=cfg.faiss_config)

    run_tag = f"{model.split('/')[-1]}__{mode}__L{max_seq}"
    return method_fn, run_tag


# bm25 builder — scope(doc|keyword). keyword 는 원본 gold KP 로딩(미지원 데이터셋이면 ValueError).
def _build_bm25(cfg):
    scope = cfg.method_params.get("scope", "doc")
    kp = None
    if scope == "keyword":
        from sdlp.methods.bm25 import load_keyphrases_by_family
        kp = load_keyphrases_by_family(cfg.prepared_dir, cfg.resolved_original_set)

    def method_fn(reference_df, query_df):
        from sdlp.methods.bm25 import bm25_votes
        return bm25_votes(reference_df, query_df, scope=scope, keyphrase_by_family=kp)

    return method_fn, f"bm25__{scope}"


# embedding_pooled builder — chunk_voting 과 같은 50단어 청크 임베딩(캐시 공유)을 문서별 mean-pool.
def _build_embedding_pooled(cfg):
    def method_fn(reference_df, query_df):
        from sdlp.methods.embedding_pooled import embedding_pooled_votes
        return embedding_pooled_votes(reference_df, query_df, cfg)

    return method_fn, f"{cfg.embed_spec.slug()}__{cfg.chunk_spec.slug()}__pooled"


# ssdeep builder — 퍼지 해시(ppdeep) 전문 비교. 파라미터 없음.
def _build_ssdeep(cfg):
    def method_fn(reference_df, query_df):
        from sdlp.methods.ssdeep import ssdeep_votes
        return ssdeep_votes(reference_df, query_df)

    return method_fn, "ssdeep"


# minhash_lsh builder — num_perm/threshold/shingle_k. LSH 인덱스로 후보 생성.
def _build_minhash_lsh(cfg):
    p = cfg.method_params
    num_perm = int(p.get("num_perm", 128))
    threshold = float(p.get("threshold", 0.2))
    shingle_k = int(p.get("shingle_k", 5))

    def method_fn(reference_df, query_df):
        from sdlp.methods.minhash_lsh import minhash_lsh_votes
        return minhash_lsh_votes(reference_df, query_df, num_perm=num_perm,
                                 threshold=threshold, shingle_k=shingle_k)

    return method_fn, f"minhash_lsh__p{num_perm}__t{threshold}__k{shingle_k}"


# method 이름 → builder(cfg) -> (method_fn, run_tag). 미구현 라이벌은 아직 등록 전.
METHOD_BUILDERS = {"longctx": _build_longctx, "bm25": _build_bm25,
                   "embedding_pooled": _build_embedding_pooled, "ssdeep": _build_ssdeep,
                   "minhash_lsh": _build_minhash_lsh}

__all__ = ["longctx_votes", "run_method", "save_method_run", "method_run_dir", "METHOD_BUILDERS"]
