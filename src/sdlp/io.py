"""파일 입출력 헬퍼 (parquet / json) 와 prepared set IO.

ponytail: yaml(config 저장) 은 S8 파이프라인에서 필요해질 때 `uv add pyyaml` 후 추가.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sdlp.schemas import normalize_docs_df


# 디렉터리를 (없으면) 만들고 Path 반환.
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# dict/list 등을 JSON 파일로 저장 (한글 보존, 직렬화 불가 값은 str 처리).
def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# JSON 파일 로드.
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# DataFrame 을 parquet 으로 저장 (index 미포함).
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# docs_df 를 스키마 정규화 후 '<set_name>.parquet' 로 저장.
def save_prepared_set(df: pd.DataFrame, prepared_dir: str | Path, set_name: str) -> Path:
    prepared_dir = ensure_dir(prepared_dir)
    out_df = normalize_docs_df(df)
    path = prepared_dir / f"{set_name}.parquet"
    out_df.to_parquet(path, index=False)
    return path


# prepared set('<set_name>.parquet') 로드 (없으면 에러).
def load_prepared_set(prepared_dir: str | Path, set_name: str) -> pd.DataFrame:
    path = Path(prepared_dir) / f"{set_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"prepared set 없음: {path}")
    return pd.read_parquet(path)


# prepared_dir 안의 모든 set 이름(확장자 제외)을 정렬해 반환.
def list_prepared_sets(prepared_dir: str | Path) -> list[str]:
    prepared_dir = Path(prepared_dir)
    if not prepared_dir.exists():
        return []
    return sorted(p.stem for p in prepared_dir.glob("*.parquet"))


# 여러 prepared set 을 합쳐 하나의 docs_df 로 반환 (doc_id 중복 제거).
# ponytail: 현재 doc_id 기준 중복 제거만. 텍스트/family 중복 점검은 S9 prepare 감사에서.
def concat_prepared_sets(prepared_dir: str | Path, set_names: list[str]) -> pd.DataFrame:
    if not set_names:
        raise ValueError("set_names 가 비어있음.")
    dfs = [load_prepared_set(prepared_dir, name) for name in set_names]
    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return normalize_docs_df(out)
