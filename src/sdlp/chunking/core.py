"""고정길이 슬라이딩 윈도우 청킹 (char / word / token 모드, overlap 지원).

논문 기본 방법은 word 고정길이(overlap=0, Kd = ceil(|d|/L)) 이지만,
ablation study 를 위해 char/token 모드와 overlap 을 함께 제공한다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from sdlp.schemas import CHUNK_META_COLUMNS


# 청킹 설정: 단위(mode), 청크 크기(size), 겹침(overlap). 캐시 키로 slug() 사용.
@dataclass(frozen=True)
class ChunkSpec:
    mode: str = "word"   # "char" | "word" | "token"
    size: int = 50
    overlap: int = 0

    # 설정 유효성 검사.
    def __post_init__(self) -> None:
        if self.mode not in {"char", "word", "token"}:
            raise ValueError(f"지원하지 않는 mode: {self.mode}")
        if self.size <= 0:
            raise ValueError("size 는 1 이상이어야 함")
        if self.overlap < 0:
            raise ValueError("overlap 은 0 이상이어야 함")
        if self.overlap >= self.size:
            raise ValueError("overlap 은 size 보다 작아야 함")

    # 캐시 키/식별용 짧은 문자열 (예: 'word50o0', 'char200o20', 'tok128o16').
    def slug(self) -> str:
        prefix = {"char": "char", "word": "word", "token": "tok"}[self.mode]
        return f"{prefix}{self.size}o{self.overlap}"


# 단위 개수 n 을 size/overlap 으로 슬라이딩해 (start, end) 구간 목록 생성.
def _sliding_windows(n_units: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if n_units <= 0:
        return []
    step = size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_units:
        end = min(start + size, n_units)
        spans.append((start, end))
        if end == n_units:
            break
        start += step
    return spans


# char 모드: 글자 단위 슬라이딩 (빈 청크 제외).
def _chunk_char(text: str, spec: ChunkSpec) -> list[str]:
    spans = _sliding_windows(len(text), spec.size, spec.overlap)
    return [p for s, e in spans if (p := text[s:e].strip())]


# word 모드: 공백 분리 단어 단위 슬라이딩 (빈 청크 제외).
def _chunk_word(text: str, spec: ChunkSpec) -> list[str]:
    words = text.split()
    spans = _sliding_windows(len(words), spec.size, spec.overlap)
    return [p for s, e in spans if (p := " ".join(words[s:e]).strip())]


# token 모드: 임베딩 모델 tokenizer 의 토큰 id 단위 슬라이딩 (빈 청크 제외).
def _chunk_token(text: str, spec: ChunkSpec, tokenizer) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    spans = _sliding_windows(len(token_ids), spec.size, spec.overlap)
    out: list[str] = []
    for s, e in spans:
        piece = tokenizer.decode(token_ids[s:e], skip_special_tokens=True).strip()
        if piece:
            out.append(piece)
    return out


# 텍스트를 spec 에 따라 청크 리스트로 자른다 (token 모드는 tokenizer 필요).
def chunk_text(text: str, spec: ChunkSpec, tokenizer=None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if spec.mode == "char":
        return _chunk_char(text, spec)
    if spec.mode == "word":
        return _chunk_word(text, spec)
    # token
    if tokenizer is None:
        raise ValueError("token 모드는 임베딩 모델 tokenizer 가 필요함")
    return _chunk_token(text, spec, tokenizer)


# docs_df 의 각 문서를 청킹해 청크 단위 DataFrame 으로 변환 (CHUNK_META_COLUMNS + chunk_text).
# 청크가 하나도 안 나오는 빈 문서는 경고 후 건너뛴다.
def build_chunks_df(docs_df: pd.DataFrame, spec: ChunkSpec, tokenizer=None) -> pd.DataFrame:
    if spec.mode == "token" and tokenizer is None:
        raise ValueError("token 모드는 임베딩 모델 tokenizer 가 필요함")

    rows: list[dict] = []
    slug = spec.slug()
    for row in docs_df.itertuples(index=False):
        chunks = chunk_text(row.text, spec, tokenizer=tokenizer)
        if not chunks:
            warnings.warn(f"청크 0개(빈 문서): {row.doc_id}")
            continue
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{row.doc_id}::c{i}",
                "doc_id": row.doc_id,
                "family_id": row.family_id,
                "dataset": row.dataset,
                "chunk_index": i,
                "chunk_len": len(chunk),
                "chunk_spec": slug,
                "chunk_text": chunk,
            })
    if not rows:
        return pd.DataFrame(columns=CHUNK_META_COLUMNS + ["chunk_text"])
    return pd.DataFrame(rows)
