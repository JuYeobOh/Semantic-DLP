"""keyphrase 데이터셋(semeval / krapivin) 원본 → 정규 docs_df.

소스: JSONL, 레코드 = {name, title, abstract, fulltext, keywords}.
text = title\n\nAbstract\n{abstract}\n\n{fulltext} (각 필드 공백 정리, keywords 는 meta 에만).
family_id = <dataset>:<name>, doc_id = <family_id>:orig.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sdlp.ids import build_doc_id, build_family_id
from sdlp.io import save_prepared_set
from sdlp.schemas import DOC_COLUMNS, normalize_docs_df


# 공백 정규화 (nbsp/개행/다중공백 → 단일 공백).
def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace(" ", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


# 한 레코드 → 문서 텍스트: title\n\nAbstract\n{abstract}\n\n{fulltext}.
def _record_to_text(rec: dict) -> str:
    title, abstract, fulltext = _clean(rec.get("title")), _clean(rec.get("abstract")), _clean(rec.get("fulltext"))
    parts: list[str] = []
    if title:
        parts += [title, ""]
    if abstract:
        parts += ["Abstract", abstract, ""]
    if fulltext:
        parts += [fulltext]
    return "\n".join(parts).strip()


# 소스 파일을 레코드 리스트로 로드 (JSON array 또는 JSONL 둘 다 지원).
def _load_records(src_path: str | Path) -> list[dict]:
    txt = Path(src_path).read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


# keyphrase 원본을 정규 docs_df 로 변환 (dataset = "semeval" | "krapivin").
def build_keyphrase_docs_df(src_path: str | Path, dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in _load_records(src_path):
        name = _clean(rec.get("name"))
        text = _record_to_text(rec)
        if not name or not text:
            continue
        family_id = build_family_id(dataset, name)
        rows.append({
            "doc_id": build_doc_id(family_id, "orig"),
            "dataset": dataset,
            "family_id": family_id,
            "text": text,
            "variant_type": "original",
            "source_doc_id": None,
            "meta_json": {"raw_name": name, "title": _clean(rec.get("title")), "keywords": _clean(rec.get("keywords"))},
        })
    if not rows:
        return pd.DataFrame(columns=DOC_COLUMNS)
    return normalize_docs_df(pd.DataFrame(rows))


# 원본 세트를 만들어 '{dataset}_original.parquet' 로 저장.
def build_and_save_keyphrase_original(src_path: str | Path, dataset: str, prepared_dir: str | Path) -> None:
    save_prepared_set(build_keyphrase_docs_df(src_path, dataset), prepared_dir, f"{dataset}_original")
