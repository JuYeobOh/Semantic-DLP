"""표준 라이벌 method 실행 harness — 검색 메커니즘이 다른 method(longctx/bm25/ssdeep 등)를
chunk_voting 과 동일한 split·평가·산출물로 감싼다.

`method_fn(reference_docs_df, query_docs_df) -> (votes_df, timing)` 만 있으면
latency·F1·confusion·케이스가 chunk_voting 과 같은 노트북 셀로 조회된다.
경량 run_dir(artifacts/method_runs/<method>__<slug>/): votes / query_manifest / per_query_eval / metrics.json.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdlp.ids import sanitize_piece
from sdlp.io import ensure_dir, save_json, save_parquet
from sdlp.metrics.core import build_query_manifest, evaluate_run
from sdlp.pipeline.config import RunConfig
from sdlp.splits.core import build_run_query_sets, make_split


# method run 산출물 디렉터리 — chunk_voting 과 동일한 통일 레이아웃 runs/{run_ident}/{run_tag}/.
# run_tag: method 고유 파라미터 leaf(예: longctx 는 "qwen3-0.6b__mean_pool__L32768__bf16").
def method_run_dir(cfg: RunConfig, method_name: str, run_tag: str = "") -> Path:
    leaf = sanitize_piece(run_tag) if run_tag else "default"
    return Path(cfg.artifacts_dir) / "runs" / cfg.run_ident(method_name) / leaf


# votes 를 평가 + 경량 run_dir 저장 → metrics 반환 (prepared IO 불필요 → 테스트 대상).
def save_method_run(run_dir, manifest_df, votes_df, timing, method_name, counts) -> dict:
    run_dir = ensure_dir(run_dir)
    metrics, eval_df = evaluate_run(manifest_df, votes_df)   # threshold=None → best-F1
    metrics["method"] = method_name
    metrics["timing_sec"] = dict(timing)   # method 가 inference_total_sec 포함해 보고(§4)
    metrics["counts"] = dict(counts)
    save_parquet(votes_df, run_dir / "votes.parquet")
    save_parquet(manifest_df, run_dir / "query_manifest.parquet")
    save_parquet(eval_df, run_dir / "per_query_eval.parquet")
    save_json(metrics, run_dir / "metrics.json")
    return metrics


# cfg 의 split·query set 을 만들고 method_fn 을 실행 → 평가 + 저장. metrics 반환.
def run_method(cfg: RunConfig, method_fn, method_name: str, run_tag: str = "") -> dict:
    split = make_split(cfg.prepared_dir, cfg.splits_dir, cfg.dataset,
                       cfg.resolved_original_set, cfg.split_seed, cfg.split_ratio)
    qs = build_run_query_sets(cfg.prepared_dir, split, cfg.resolved_original_set,
                              cfg.resolved_variant_sets, cfg.include_original_as_positive)
    manifest = build_query_manifest(qs.positive_df, qs.benign_df)
    query_df = pd.concat([qs.positive_df, qs.benign_df], ignore_index=True)

    votes, timing = method_fn(qs.reference_df, query_df)
    counts = {
        "n_ref_docs": int(len(qs.reference_df)),
        "n_query_docs": int(len(qs.positive_df) + len(qs.benign_df)),
        "n_positive": int(len(qs.positive_df)),
        "n_benign": int(len(qs.benign_df)),
    }
    return save_method_run(method_run_dir(cfg, method_name, run_tag),
                           manifest, votes, timing, method_name, counts)
