"""대안 method 패키지 — chunk_voting 과 동일한 votes_df 를 만드는 drop-in 구현들.

성격 라벨(제안/baseline) 없이 이름만. 분류는 리포트 단계 category 로만 정한다.
각 method: xxx_votes(reference_docs_df, query_docs_df, ...) -> (votes_df, timing).
votes_df 는 sdlp.voting.core.VOTES_COLUMNS 호환(confidence 컬럼) → 같은 evaluate_run 으로 평가.
run_method 로 감싸면 경량 run_dir(votes/manifest/per_query_eval/metrics.json)에 저장 →
chunk_voting 과 동일한 latency·F1·confusion·케이스 조회 셀 재사용.
"""
from sdlp.methods.longctx import longctx_votes
from sdlp.methods.runner import method_run_dir, run_method, save_method_run

__all__ = ["longctx_votes", "run_method", "save_method_run", "method_run_dir"]
