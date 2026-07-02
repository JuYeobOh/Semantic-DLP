"""평가 지표 — detection(P/R/F1/Acc) + family attribution + threshold-independent PR-AUC/ROC-AUC.

timing/throughput/counts 는 여기서 계산하지 않는다 (S8 파이프라인이 metrics dict 에 합침).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sdlp.detection.core import apply_threshold
from sdlp.io import save_json


# 0 나눗셈 방지 나눗셈.
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# 쿼리 정답표: positive 는 자기 family 가 정답(target), benign 은 정답 없음(None).
def build_query_manifest(positive_docs_df: pd.DataFrame, benign_docs_df: pd.DataFrame) -> pd.DataFrame:
    pos = [
        {"query_doc_id": r.doc_id, "is_positive": True, "target_family_id": r.family_id}
        for r in positive_docs_df.itertuples(index=False)
    ]
    neg = [
        {"query_doc_id": r.doc_id, "is_positive": False, "target_family_id": None}
        for r in benign_docs_df.itertuples(index=False)
    ]
    return pd.DataFrame(pos + neg)


# votes 에 임계값 적용 후 manifest 와 대조 → (집계 metrics dict, per-query merged df).
def evaluate_run(
    query_manifest_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    threshold: float,
    confidence_col: str = "confidence",
) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = apply_threshold(votes_df, threshold=threshold, confidence_col=confidence_col)
    merged = query_manifest_df.merge(pred, on="query_doc_id", how="left")
    merged["pred_detected"] = merged["pred_detected"].fillna(False)

    yt = merged["is_positive"].astype(int).to_numpy()
    yp = merged["pred_detected"].astype(int).to_numpy()
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    # attribution: positive 쿼리가 올바른 family 로 탐지됐는지
    pos_mask = merged["is_positive"] == True
    det_pos_mask = pos_mask & (merged["pred_detected"] == True)
    fam_ok = merged["pred_family_id"] == merged["target_family_id"]
    attr_all = _safe_div(int((det_pos_mask & fam_ok).sum()), int(pos_mask.sum()))
    attr_detected = _safe_div(int((det_pos_mask & fam_ok).sum()), int(det_pos_mask.sum()))

    # threshold-independent 분리도 (미탐 confidence 는 0 으로). 양성·음성 둘 다 있어야 정의됨.
    score = merged[confidence_col].fillna(0.0).to_numpy()
    label = yt
    pr_auc = roc_auc = None
    if len(label) and 0 < label.sum() < len(label):
        pr_auc = float(average_precision_score(label, score))
        roc_auc = float(roc_auc_score(label, score))

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "confidence_col": confidence_col,
        "detection": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        },
        "attribution": {
            "family_acc_on_all_positive": attr_all,
            "family_acc_on_detected_positive": attr_detected,
        },
        "separability": {
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "positive_ratio": float(label.mean()) if len(label) else 0.0,
        },
    }
    return metrics, merged


# family_id 별 detection/attribution 진단표 (어느 기밀 문서가 잘 탐지/오탐되나).
def build_errors_by_family(eval_df: pd.DataFrame) -> pd.DataFrame:
    pos = eval_df[eval_df["is_positive"] == True].copy()
    neg = eval_df[eval_df["is_positive"] == False].copy()

    pos["_detected"] = (pos["pred_detected"] == True).astype(int)
    pos["_attr_ok"] = (
        (pos["pred_detected"] == True) & (pos["pred_family_id"] == pos["target_family_id"])
    ).astype(int)
    pos_grp = pos.groupby("target_family_id", dropna=True).agg(
        n_positive_total=("query_doc_id", "size"),
        n_positive_detected=("_detected", "sum"),
        n_attribution_correct=("_attr_ok", "sum"),
    )

    # benign 이 특정 family 로 잘못 끌려간 횟수(false attraction)
    neg_det = neg[neg["pred_detected"] == True]
    if len(neg_det):
        neg_grp = neg_det.groupby("pred_family_id", dropna=True).size().rename("n_false_attraction").to_frame()
    else:
        neg_grp = pd.DataFrame(columns=["n_false_attraction"])

    out = pos_grp.join(neg_grp, how="outer").fillna(0).astype(int)
    out.index.name = "family_id"
    out = out.reset_index()
    out["detection_rate"] = out.apply(lambda r: _safe_div(r["n_positive_detected"], r["n_positive_total"]), axis=1)
    out["attribution_rate"] = out.apply(lambda r: _safe_div(r["n_attribution_correct"], r["n_positive_detected"]), axis=1)
    return out.sort_values(["n_positive_total", "n_false_attraction"], ascending=[False, False]).reset_index(drop=True)


# benign(비기밀) 쿼리인데 탐지된 = 오탐(FP). 어느 family/문서로 잘못 끌렸는지 쌍으로 반환 (오탐 분석용).
# confidence 내림차순 정렬 → 가장 심한 오탐부터.
def build_false_positive_pairs(eval_df: pd.DataFrame) -> pd.DataFrame:
    fp = eval_df[(eval_df["is_positive"] == False) & (eval_df["pred_detected"] == True)].copy()
    cols = [
        "query_doc_id", "query_family_id", "pred_doc_id", "pred_family_id",
        "confidence", "best_votes", "n_chunks", "vote_entropy", "vote_gini",
        "vote_distribution_json",
    ]
    cols = [c for c in cols if c in fp.columns]   # eval_df 에 있는 컬럼만 (호환)
    return fp[cols].sort_values("confidence", ascending=False).reset_index(drop=True)


# metrics dict 를 JSON 으로 저장.
def save_metrics_json(metrics: dict, path) -> None:
    save_json(metrics, path)
