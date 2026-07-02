"""openset 판별 — confidence 임계값 결정 + 투표 분포 보조통계(entropy/variance/Gini).

핵심 직관: 쿼리가 기밀이면 top-1 투표가 한 문서에 집중(엔트로피·분산 낮음),
비기밀이면 여러 문서로 분산(높음). 주 지표 confidence = 최다 득표수 / 전체 청크 수.
보조통계는 이 분포의 집중도를 다른 각도로 재는 값들.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 투표 수 배열을 확률분포로 정규화 (합 0이면 그대로).
def _to_probs(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


# 투표 분포의 정규화 엔트로피 [0,1] (0=한 문서 집중, 1=고르게 분산). 득표 문서 1개면 0.
def vote_entropy(counts) -> float:
    probs = _to_probs(counts)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / np.log(len(probs))


# 투표 분포의 Gini 불순도 1 - Σp² (0=한 문서 집중, →1 분산).
def vote_gini(counts) -> float:
    probs = _to_probs(counts)
    return float(1.0 - np.sum(probs ** 2))


# 투표 확률값들의 분산 (한 문서에 집중될수록 큼).
def vote_variance(counts) -> float:
    probs = _to_probs(counts)
    return float(np.var(probs)) if len(probs) else 0.0


# confidence 가 threshold 미만이면 미탐(pred None) 처리. pred_detected 컬럼 추가한 복사본 반환.
def apply_threshold(
    votes_df: pd.DataFrame, threshold: float, confidence_col: str = "confidence"
) -> pd.DataFrame:
    out = votes_df.copy()
    if len(out) == 0:
        out["pred_detected"] = pd.Series(dtype=bool)
        return out
    out["pred_detected"] = out[confidence_col] >= threshold
    low = ~out["pred_detected"]
    out.loc[low, "pred_doc_id"] = None
    out.loc[low, "pred_family_id"] = None
    return out
