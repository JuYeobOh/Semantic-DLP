"""문서표(docs_df)·청크표의 정규 스키마.

- prepared parquet 은 모두 ``DOC_COLUMNS`` 스키마를 따른다.
- 청크 메타 parquet 은 ``CHUNK_META_COLUMNS`` 스키마를 따른다.
"""
from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

# 한 문서 = 한 행. 모든 prepared parquet 이 이 컬럼 순서를 가진다.
DOC_COLUMNS: list[str] = [
    "doc_id",         # 문서 고유 id
    "dataset",        # 데이터셋 이름 (semeval / krapivin / re3 / casimir / par3)
    "family_id",      # 같은 원문에서 파생된 문서들을 묶는 id (50/50 분할 단위)
    "text",           # 문서 본문
    "version_index",  # 리비전 순서 (원본=1, 이후 2,3...). 변형셋은 비어있을 수 있음
    "variant_type",   # original | revision | eda | gpt_para | dipper | ...
    "variant_level",  # 변형 강도 (예: eda alpha)
    "variant_seed",   # 변형 재현용 seed
    "source_doc_id",  # 이 문서가 파생된 원본 doc_id (정답 attribution 용)
    "num_chars",      # 본문 글자 수 (자동 계산)
    "num_words",      # 본문 단어 수 (자동 계산)
    "meta_json",      # 데이터셋별 부가정보 (JSON 문자열)
]

# 한 청크 = 한 행 (본문 text 는 별도 컬럼으로 append 됨).
CHUNK_META_COLUMNS: list[str] = [
    "chunk_id",
    "doc_id",
    "family_id",
    "dataset",
    "chunk_index",
    "chunk_len",
    "chunk_spec",
]


# df 에 required 컬럼이 모두 있는지 검사하고, 없으면 에러.
def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} 에 필수 컬럼 누락: {missing}")


# meta_json 값을 항상 JSON 문자열로 통일 (None→'{}', dict→직렬화).
def _to_meta_json(value: object) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# 임의의 docs_df 를 DOC_COLUMNS 스키마로 정규화한다.
# - 필수 컬럼(doc_id/dataset/family_id/text) 없으면 에러
# - 선택 컬럼은 기본값으로 채우고 num_chars/num_words 자동 계산
# - doc_id 중복이면 에러
def normalize_docs_df(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["doc_id", "dataset", "family_id", "text"], "docs_df")

    out = df.copy()

    # 선택 컬럼 기본값
    optional_defaults: dict[str, object] = {
        "version_index": pd.NA,
        "variant_type": "original",
        "variant_level": pd.NA,
        "variant_seed": pd.NA,
        "source_doc_id": pd.NA,
        "meta_json": "{}",
    }
    for col, default_value in optional_defaults.items():
        if col not in out.columns:
            out[col] = default_value

    out["text"] = out["text"].fillna("").astype(str)
    out["num_chars"] = out["text"].map(len)
    out["num_words"] = out["text"].map(lambda x: len(x.split()))
    out["meta_json"] = out["meta_json"].map(_to_meta_json)

    dup_mask = out["doc_id"].duplicated()
    if dup_mask.any():
        dupes = out.loc[dup_mask, "doc_id"].tolist()[:10]
        raise ValueError(f"docs_df 에 중복 doc_id: {dupes}")

    for col in DOC_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[DOC_COLUMNS].reset_index(drop=True)
