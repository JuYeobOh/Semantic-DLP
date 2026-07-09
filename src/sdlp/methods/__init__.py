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
    dtype = p.get("dtype", "float32")
    batch = int(p.get("batch_size", 8))
    orig, seed = cfg.resolved_original_set, cfg.split_seed
    q_tag = f"{orig}__s{seed}" + ("__inclorig" if cfg.include_original_as_positive else "")

    def method_fn(reference_df, query_df):
        return longctx_votes(
            reference_df, query_df, long_doc=mode, model_name=model, max_seq_length=max_seq,
            dtype=dtype, batch_size=batch, artifacts_dir=cfg.artifacts_dir,
            ref_set_name=f"{orig}__s{seed}__ref", query_set_name=f"{q_tag}__query")

    run_tag = f"{model.split('/')[-1]}__{mode}__L{max_seq}"
    return method_fn, run_tag


# method 이름 → builder(cfg) -> (method_fn, run_tag). 미구현 라이벌은 아직 등록 전.
METHOD_BUILDERS = {"longctx": _build_longctx}

__all__ = ["longctx_votes", "run_method", "save_method_run", "method_run_dir", "METHOD_BUILDERS"]
