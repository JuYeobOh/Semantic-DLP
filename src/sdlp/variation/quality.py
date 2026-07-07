"""변형 품질 지표 — 어휘 변형(변형 강도) 측면. (의미 보존 지표는 이후 추가)

(원본 문서, 변형 문서) 쌍에 대해 문서 단위로 계산. 방향:
- ngram_overlap(1/2/3) / self_bleu / jaccard_tokens : **낮을수록 변형 강함**(원문 표현 덜 유지).
- norm_edit_distance : **높을수록 변형 강함**.

토큰화는 소문자 [a-z0-9]+ 로 단순·결정적. 전부 순수 함수(외부 모델 불필요) → 결정적 단위 테스트.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein


# 소문자 영숫자 토큰 목록.
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


# 토큰열의 n-gram 튜플 목록 (길이 부족 시 빈 목록).
def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# n-gram overlap: 변형의 distinct n-gram 중 원본에도 있는 비율(정밀도). 낮을수록 변형↑.
def ngram_overlap(ref: str, hyp: str, n: int = 1) -> float:
    ref_ng = set(_ngrams(_tokenize(ref), n))
    hyp_ng = set(_ngrams(_tokenize(hyp), n))
    if not hyp_ng:
        return 0.0
    return len(ref_ng & hyp_ng) / len(hyp_ng)


# 토큰 집합 Jaccard 유사도(교집합/합집합). 낮을수록 변형↑.
def jaccard_tokens(ref: str, hyp: str) -> float:
    ref_set, hyp_set = set(_tokenize(ref)), set(_tokenize(hyp))
    union = ref_set | hyp_set
    if not union:
        return 0.0
    return len(ref_set & hyp_set) / len(union)


# 토큰 단위 정규화 편집거리(거리/최대길이). 높을수록 변형↑.
# rapidfuzz(C 가속) — 전문(수천 토큰) 순수 파이썬 O(n²)는 수십초라 실용 불가.
def norm_edit_distance(ref: str, hyp: str) -> float:
    r, h = _tokenize(ref), _tokenize(hyp)
    if not r and not h:
        return 0.0
    return Levenshtein.distance(r, h) / max(len(r), len(h))


# self-BLEU: 변형(hyp)을 원본(ref) 기준으로 잰 BLEU(최대 4-gram, +1 스무딩, brevity penalty). 낮을수록 변형↑.
def self_bleu(ref: str, hyp: str, max_n: int = 4) -> float:
    ref_toks, hyp_toks = _tokenize(ref), _tokenize(hyp)
    if not hyp_toks:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp_toks, n))
        if not hyp_ng:                       # hyp 이 n 보다 짧으면 상위 n 중단
            break
        ref_ng = Counter(_ngrams(ref_toks, n))
        overlap = sum(min(c, ref_ng.get(gram, 0)) for gram, c in hyp_ng.items())
        precisions.append((overlap + 1) / (sum(hyp_ng.values()) + 1))
    if not precisions:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    brevity = 1.0 if len(hyp_toks) >= len(ref_toks) else math.exp(1 - len(ref_toks) / len(hyp_toks))
    return brevity * geo_mean


# (원본, 변형) 한 쌍의 어휘 변형 지표 전부.
def lexical_metrics(ref: str, hyp: str) -> dict[str, float]:
    return {
        "ngram_overlap_1": ngram_overlap(ref, hyp, 1),
        "ngram_overlap_2": ngram_overlap(ref, hyp, 2),
        "ngram_overlap_3": ngram_overlap(ref, hyp, 3),
        "self_bleu": self_bleu(ref, hyp),
        "norm_edit_distance": norm_edit_distance(ref, hyp),
        "jaccard_tokens": jaccard_tokens(ref, hyp),
    }


# 변형셋 각 문서를 source_doc_id 로 원본과 짝지어 어휘 지표 DataFrame 생성 (문서 단위 행).
def pairwise_lexical_df(original_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    ref_text = dict(zip(original_df["doc_id"], original_df["text"]))
    rows = []
    for row in variant_df.itertuples(index=False):
        ref = ref_text.get(row.source_doc_id)
        if ref is None:                      # 원본 없는 변형은 건너뜀
            continue
        rows.append({
            "doc_id": row.doc_id,
            "source_doc_id": row.source_doc_id,
            "family_id": row.family_id,
            "variant_type": row.variant_type,
            "variant_level": row.variant_level,
            **lexical_metrics(ref, row.text),
        })
    return pd.DataFrame(rows)
