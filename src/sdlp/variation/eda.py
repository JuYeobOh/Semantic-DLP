"""EDA synonym-replacement(sr) 변형 생성 (Wei & Zou 2019 의 sr 기법만).

원본 docs_df → 각 문서의 단어 alpha 비율을 WordNet 동의어로 치환한 변형 docs_df.
- variant_type=eda, variant_level=alpha, variant_seed=seed
- doc_id={family}:eda-sr@{alpha}:s{seed}, set 이름 {dataset}_original_eda_sr_a{NN}_s{seed}

ponytail: sr 만 이식(rd/ri/rs 미사용). 동의어원(synonyms_fn)은 주입식 → 테스트는 nltk/WordNet 불필요.
난수: 모듈 전역 random 을 seed(=42)로 1회 초기화 후 문서 순서대로 상태를 이어받는다(원 EDA 구조).
     → 배치(build) 단위로 결정적. 원본 parquet 은 seed 1 이라 다르므로 이 코드로 재생성해야 함.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from sdlp.ids import build_doc_id
from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# EDA 원본 불용어(치환 후보에서 제외).
STOP_WORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now",
})

# 동의어 문자로 허용하는 문자셋 (원 EDA get_synonyms 와 동일).
_SYNONYM_CHARS = set(" qwertyuiopasdfghjklzxcvbnm")


# 소문자화 + 영숫자/일부기호(<>=,.:-)만 남기고 공백 정리 (원 EDA get_only_chars).
def clean_text(line: str) -> str:
    line = line.replace("'", "").replace("'", "")
    line = line.replace("\t", " ").replace("\n", " ").lower()
    line = re.sub(r"[^a-z0-9 <>=,.:\-]", " ", line)
    line = re.sub(r"\s*,\s*", " , ", line)
    return re.sub(r"\s+", " ", line).strip()


# "word , next" → "word, next" (쉼표 주변 공백 정리).
def _detokenize_commas(text: str) -> str:
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r",\s*", ", ", text).strip()


# WordNet 동의어 리스트 (nltk 지연 import; 최초 1회 nltk.download('wordnet','omw-1.4') 필요).
def wordnet_synonyms(word: str) -> list[str]:
    from nltk.corpus import wordnet

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            s = lemma.name().replace("_", " ").replace("-", " ").lower()
            s = "".join(c for c in s if c in _SYNONYM_CHARS)
            synonyms.add(s)
    synonyms.discard(word)
    return list(synonyms)


# 치환 후보 판정: 알파벳 단어이며 불용어가 아님.
def _is_candidate(token: str) -> bool:
    return token.isalpha() and token not in STOP_WORDS


# words 중 최대 n 개를 동의어로 치환 (모듈 전역 random 으로 후보 셔플·동의어 선택). 동의어 없는 단어는 건너뜀.
def synonym_replacement(words: list[str], n: int, synonyms_fn) -> list[str]:
    candidates = list({w for w in words if _is_candidate(w)})
    random.shuffle(candidates)
    new_words = list(words)
    replaced = 0
    for word in candidates:
        synonyms = synonyms_fn(word)
        if synonyms:
            choice = random.choice(list(synonyms))
            new_words = [choice if w == word else w for w in new_words]
            replaced += 1
        if replaced >= n:
            break
    return new_words


# 한 문서 텍스트를 EDA-sr 로 변형 (단어의 alpha 비율 치환).
# ⚠️ 전역 random 상태 사용 → 배치 재현은 build 가 random.seed 로 담당(단독 호출 시 앞서 random.seed 필요).
def eda_sr(text: str, alpha: float, synonyms_fn=wordnet_synonyms) -> str:
    words = [w for w in clean_text(text).split(" ") if w]
    if not words:
        return ""
    n = max(1, int(alpha * len(words)))
    new_words = synonym_replacement(words, n, synonyms_fn)
    return _detokenize_commas(" ".join(new_words))


# 원본 docs_df → EDA-sr 변형 docs_df (스키마 정합).
def build_eda_variant_df(
    original_df: pd.DataFrame, alpha: float, seed: int = 42, synonyms_fn=wordnet_synonyms,
) -> pd.DataFrame:
    random.seed(seed)   # 전역 시드 1회 — 이후 문서들이 상태를 이어받아 배치 전체가 결정적(원 EDA 방식).
    doc_key = f"eda-sr@{alpha}:s{seed}"
    rows = []
    for row in original_df.itertuples(index=False):
        rows.append({
            "doc_id": build_doc_id(row.family_id, doc_key),
            "dataset": row.dataset,
            "family_id": row.family_id,
            "text": eda_sr(row.text, alpha, synonyms_fn),
            "variant_type": "eda",
            "variant_level": alpha,
            "variant_seed": seed,
            "source_doc_id": row.doc_id,
            "meta_json": {"eda_technique": "sr", "alpha": alpha, "seed": seed},
        })
    return normalize_docs_df(pd.DataFrame(rows))


# 변형 세트 저장: {dataset}_original_eda_sr_a{NN}_s{seed}.
def build_and_save_eda_variant_set(
    original_df: pd.DataFrame, prepared_dir, dataset: str, alpha: float,
    seed: int = 42, synonyms_fn=wordnet_synonyms,
) -> None:
    df = build_eda_variant_df(original_df, alpha, seed, synonyms_fn)
    name = f"{dataset}_original_eda_sr_a{int(round(alpha * 100)):02d}_s{seed}"
    save_prepared_set(df, prepared_dir, name)
