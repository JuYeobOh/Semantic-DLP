"""Re3-Sci 논문 리비전 데이터셋 → 정규 docs_df.

소스: Re3-Sci_v1.zip 안의 docs/<paper_id>/v1.json, v2.json (nodes 구조).
문서 텍스트 = KEEP 노드(article-title/abstract/title/p) content 를 순서대로 이어붙임.
v1 = 원본(re3_v1_only), v2 = 리비전(re3_v2_only). 같은 논문이 한 family.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from sdlp.ids import build_doc_id, build_family_id, make_revision_doc_key
from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# 문서 텍스트에 포함할 노드 타입 (본문 구조만; ref/fig/table 등은 제외).
KEEP_NTYPES = {"article-title", "abstract", "title", "p"}
# zip 안의 문서 파일 경로 패턴: .../docs/<paper_id>/v<1|2>.json
_DOC_RE = re.compile(r"/docs/([^/]+)/v([12])\.json$")


# nodes JSON 을 문서 텍스트로 복원 (abstract 는 'Abstract' 헤더를 살림).
def json_to_text(data: dict) -> str:
    parts: list[str] = []
    for node in data.get("nodes", []):
        ntype = node.get("ntype")
        content = (node.get("content") or "").strip()
        if ntype not in KEEP_NTYPES or not content:
            continue
        if ntype == "abstract":
            parts += ["Abstract", content, ""]
        else:
            parts += [content, ""]
    return "\n".join(parts).strip()


# Re3-Sci zip 에서 paper 별 v1/v2 를 읽어 정규 docs_df 로 변환.
def build_re3_docs_df(zip_path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        versions: dict[str, dict[int, dict]] = {}
        for name in z.namelist():
            m = _DOC_RE.search(name)
            if m:
                versions.setdefault(m.group(1), {})[int(m.group(2))] = json.loads(z.read(name))

    rows: list[dict] = []
    for paper_id in sorted(versions):
        family_id = build_family_id("re3sci", paper_id)
        for version_index in (1, 2):
            data = versions[paper_id].get(version_index)
            if data is None:
                continue
            rows.append({
                "doc_id": build_doc_id(family_id, make_revision_doc_key(version_index)),
                "dataset": "re3sci",
                "family_id": family_id,
                "text": json_to_text(data),
                "version_index": version_index,
                "variant_type": "original" if version_index == 1 else "revision",
                "source_doc_id": build_doc_id(family_id, "v1") if version_index == 2 else None,
                "meta_json": {"raw_name": paper_id},
            })
    return normalize_docs_df(pd.DataFrame(rows))


# re3_v1_only(원본) / re3_v2_only(리비전) 두 세트로 저장.
def build_and_save_re3_prepared_sets(zip_path: str | Path, prepared_dir: str | Path) -> None:
    docs = build_re3_docs_df(zip_path)
    save_prepared_set(docs[docs["version_index"] == 1].reset_index(drop=True), prepared_dir, "re3_v1_only")
    save_prepared_set(docs[docs["version_index"] == 2].reset_index(drop=True), prepared_dir, "re3_v2_only")
