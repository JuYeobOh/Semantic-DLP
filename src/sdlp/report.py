"""결과 스캔 — runs/**/metrics.json 을 모아 하나의 표로. summary parquet 대신 이걸 쓴다.

레이아웃: artifacts/runs/{run_ident}/{leaf}/metrics.json. 각 metrics.json 은 dataset·method 를 담는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# artifacts/runs 아래 모든 metrics.json 을 평평한 DataFrame 으로 (method×dataset 피벗 원천).
def load_all_metrics(artifacts_dir: str | Path = "artifacts") -> pd.DataFrame:
    rows: list[dict] = []
    for mp in sorted(Path(artifacts_dir, "runs").glob("*/*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        d = m.get("detection", {})
        b = m.get("detection_best", {})
        s = m.get("separability", {})
        t = m.get("timing_sec", {})
        rows.append({
            "method": m.get("method"),
            "dataset": m.get("dataset"),
            "run_ident": mp.parent.parent.name,
            "config": mp.parent.name,
            "f1": d.get("f1"),
            "best_f1": b.get("best_f1"),
            "best_threshold": b.get("best_threshold"),
            "roc_auc": s.get("roc_auc"),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "inference_sec": t.get("inference_total_sec"),
        })
    return pd.DataFrame(rows)


# method × dataset 피벗 (기본 best_f1). rows_order/cols_order 로 표 순서 고정.
def pivot_table(df: pd.DataFrame, value: str = "best_f1",
                rows_order: list[str] | None = None, cols_order: list[str] | None = None) -> pd.DataFrame:
    pv = df.pivot_table(index="method", columns="dataset", values=value, aggfunc="max")
    if rows_order:
        pv = pv.reindex([r for r in rows_order if r in pv.index])
    if cols_order:
        pv = pv.reindex(columns=[c for c in cols_order if c in pv.columns])
    return pv


# 논문 최종 표의 행 정의 — (표시라벨, method, config leaf 부분매칭). config_sub=None 이면 method 전체.
# 같은 method 의 변형을 leaf 로 구분: chunk_voting cb/sw, longctx qwen/granite, bm25 doc/keyword.
PAPER_ROWS: list[tuple[str, str, str | None]] = [
    ("chunk + voting (Count-based)",      "chunk_voting",     "__cb"),
    ("chunk + voting (Score-weighted)",   "chunk_voting",     "__sw"),
    ("longcontext - qwen 0.6B (decoder)", "longctx",          "qwen"),
    ("longcontext - ibm 97m (encoder)",   "longctx",          "granite"),
    ("embedding mean pooling",            "embedding_pooled", None),
    ("colbertv2",                         "colbert",          None),
    ("chunk(maxsim θ)",                   "chunk_maxsim",     None),
    ("bm25 (full doc)",                   "bm25",             "bm25__doc"),
    ("bm25 (keyword)",                    "bm25",             "bm25__keyword"),
    ("ssdeep",                            "ssdeep",           None),
    ("minhash+lsh",                       "minhash_lsh",      None),
]


# 논문 최종 표 — 행 = rows_spec(변형까지 구분·순서 고정), 열 = dataset. 안 돌린 셀은 NaN.
# method 만으로 피벗하면 bm25 doc/keyword·voting cb/sw 가 뭉개지므로 config leaf 로 변형을 분리한다.
def paper_table(df: pd.DataFrame, value: str = "best_f1",
                rows_spec: list[tuple[str, str, str | None]] = PAPER_ROWS,
                cols_order: list[str] | None = None) -> pd.DataFrame:
    datasets = cols_order or sorted(df["dataset"].dropna().unique())
    out = pd.DataFrame(index=[label for label, _, _ in rows_spec], columns=datasets, dtype=float)
    for label, method, config_sub in rows_spec:
        sub = df[df["method"] == method]
        if config_sub:
            sub = sub[sub["config"].str.contains(config_sub, case=False, regex=False, na=False)]
        for ds in datasets:
            vals = sub.loc[sub["dataset"] == ds, value].dropna()
            if len(vals):
                out.loc[label, ds] = float(vals.max())
    return out
