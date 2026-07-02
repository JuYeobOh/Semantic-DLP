"""family-aware 50/50 기밀 분할.

핵심 불변식: 같은 family_id 의 모든 문서(원본·변형·버전)는 항상 같은 partition
(confidential 또는 benign). → 학습/평가 누수 차단. seed 고정으로 재현.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sdlp.io import (
    concat_prepared_sets,
    ensure_dir,
    load_prepared_set,
    save_parquet,
)

SPLIT_COLUMNS = ["family_id", "partition", "dataset", "seed"]
PARTITIONS = ("confidential", "benign")


# original_set 의 family 를 셔플해 ratio 만큼 confidential 로 분할. 파일 있으면 로드(overwrite=False).
def make_split(
    prepared_dir: str | Path,
    splits_dir: str | Path,
    dataset: str,
    original_set: str,
    seed: int = 42,
    ratio: float = 0.5,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio 는 (0,1) 여야 함: {ratio}")

    out_path = Path(splits_dir) / f"{dataset}_s{seed}.parquet"
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)

    docs = load_prepared_set(prepared_dir, original_set)
    families = np.array(sorted(docs["family_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    n_conf = int(round(len(families) * ratio))
    confidential = set(families[:n_conf].tolist())

    split_df = pd.DataFrame({
        "family_id": families,
        "partition": ["confidential" if f in confidential else "benign" for f in families],
        "dataset": dataset,
        "seed": int(seed),
    })[SPLIT_COLUMNS]

    ensure_dir(splits_dir)
    save_parquet(split_df, out_path)
    return split_df


# docs_df 에서 split 의 해당 partition 에 속하는 family 만 남긴다.
def apply_split(docs_df: pd.DataFrame, split_df: pd.DataFrame, partition: str) -> pd.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError(f"partition 은 {PARTITIONS} 중 하나여야 함: {partition!r}")
    target = set(split_df.loc[split_df["partition"] == partition, "family_id"])
    return docs_df[docs_df["family_id"].isin(target)].reset_index(drop=True)


# variant set 의 family 가 split(원본 family) 에 다 있는지 검사. strict 면 에러, 아니면 경고.
def _assert_family_coverage(docs_df, split_df, set_name, strict=False) -> None:
    missing = set(docs_df["family_id"].unique()) - set(split_df["family_id"].unique())
    if missing:
        msg = f"[splits] {set_name}: split 에 없는 family {len(missing)}개 (예: {sorted(missing)[:5]})"
        if strict:
            raise ValueError(msg)
        print(f"WARNING {msg}")


# 한 실행의 reference / positive / benign 문서 집합.
@dataclass(frozen=True)
class RunQuerySets:
    reference_df: pd.DataFrame   # 기밀 측 원본 (= 등록되는 기밀 DB)
    positive_df: pd.DataFrame    # 기밀 측 변형 (+옵션: 기밀 원본) = 유출 시도
    benign_df: pd.DataFrame      # 비기밀 측 원본 + 변형


# split 을 적용해 reference/positive/benign 을 만든다.
# include_original_as_positive=True 면 기밀 원본도 positive 쿼리로 추가 (verbatim 유출 실험).
def build_run_query_sets(
    prepared_dir: str | Path,
    split_df: pd.DataFrame,
    original_set: str,
    variant_sets: list[str],
    include_original_as_positive: bool = False,
    strict: bool = False,
) -> RunQuerySets:
    original_df = load_prepared_set(prepared_dir, original_set)
    _assert_family_coverage(original_df, split_df, original_set, strict=strict)

    if variant_sets:
        variants_df = concat_prepared_sets(prepared_dir, variant_sets)
        _assert_family_coverage(variants_df, split_df, "+".join(variant_sets), strict=strict)
    else:
        variants_df = original_df.iloc[0:0].copy()

    reference_df = apply_split(original_df, split_df, "confidential")

    positive_parts = [apply_split(variants_df, split_df, "confidential")]
    if include_original_as_positive:
        positive_parts.append(apply_split(original_df, split_df, "confidential"))
    positive_df = pd.concat(positive_parts, ignore_index=True).drop_duplicates(subset=["doc_id"])

    benign_df = pd.concat(
        [apply_split(original_df, split_df, "benign"), apply_split(variants_df, split_df, "benign")],
        ignore_index=True,
    ).drop_duplicates(subset=["doc_id"])

    return RunQuerySets(
        reference_df=reference_df.reset_index(drop=True),
        positive_df=positive_df.reset_index(drop=True),
        benign_df=benign_df.reset_index(drop=True),
    )


# split 통계 (노트북/테스트 표시용).
def split_summary(split_df: pd.DataFrame) -> dict:
    counts = split_df["partition"].value_counts().to_dict()
    return {
        "dataset": split_df["dataset"].iloc[0] if len(split_df) else None,
        "seed": int(split_df["seed"].iloc[0]) if len(split_df) else None,
        "n_families": int(len(split_df)),
        "n_confidential": int(counts.get("confidential", 0)),
        "n_benign": int(counts.get("benign", 0)),
    }
