"""doc_id / family_id 생성 규칙.

ponytail: 지금 쓰는 id 만. eda/gpt/dipper 등 변형 doc_key 는 S10 변형 모듈에서 추가.
(쿼리는 doc_id 로 유일하게 식별되므로 별도 query_id 불필요.)
"""
from __future__ import annotations

import re


# id 조각을 안전한 형태로 정규화 (소문자·공백→_·허용문자 외 제거).
def sanitize_piece(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-zA-Z0-9_@.\-:]+", "", value)
    return value


# 같은 원문 묶음(family)의 id 생성: '<dataset>:<family_key>'.
def build_family_id(dataset: str, family_key: str) -> str:
    return f"{sanitize_piece(dataset)}:{sanitize_piece(family_key)}"


# family 안의 한 문서 id 생성: '<family_id>:<doc_key>'.
def build_doc_id(family_id: str, doc_key: str) -> str:
    return f"{sanitize_piece(family_id)}:{sanitize_piece(doc_key)}"


# 리비전 문서의 doc_key: 'v1', 'v2', ... (casimir/re3 버전 문서용).
def make_revision_doc_key(version_index: int) -> str:
    return f"v{int(version_index)}"
