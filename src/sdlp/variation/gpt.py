"""GPT paraphrase 변형 생성 (의미 보존 재서술).

원본 docs_df → LLM 이 의미를 보존하며 재서술한 변형 docs_df.
- variant_type=gpt_paraphrase, variant_level=temperature, variant_seed=seed
- doc_id={family}:gpt-para@{model}:s{seed}, set 이름 {dataset}_original_gpt_para_{model_tag}_t{temp}_s{seed}

§8: 기존 gpt parquet 재생성 안 함. 이 모듈은 재현·재사용 인터페이스.
LLM 호출은 paraphrase_fn(text)->str 주입식 → 테스트는 mock, 실호출은 사용자(비쌈, 배치 권장).
"""
from __future__ import annotations

import pandas as pd

from sdlp.ids import build_doc_id
from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# 의미 보존 재서술 지시(정본 프롬프트). 정보 가감 없이 표현/구조만 바꾼다.
PARAPHRASE_SYSTEM_PROMPT = (
    "You are a careful academic paraphraser. Rewrite the given document so the wording and "
    "sentence structure differ substantially, while preserving the exact meaning, technical "
    "content, and every factual claim. Keep the same language (English) and keep section "
    "structure (e.g. an 'Abstract' heading). Do not add, remove, summarize, or reorder "
    "information, and do not add any commentary. Return only the rewritten document text."
)


# 모델 id → set 이름 태그 (gpt-5.4 → 54).
def _model_tag(model: str) -> str:
    return model.replace("gpt-", "").replace(".", "").replace("-", "")


# OpenAI 로 한 문서를 재서술하는 callable(text)->str 생성 (키는 client 가 env 에서 읽음).
# 지연 import — 테스트/EDA 경로는 openai 불필요. 실호출은 사용자.
def openai_paraphrase_fn(model: str = "gpt-5.4", temperature: float = 0, seed: int = 42):
    from openai import OpenAI

    client = OpenAI()  # OPENAI_API_KEY 환경변수 자동 사용

    def paraphrase(text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            messages=[
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content or ""

    return paraphrase


# 원본 docs_df → GPT paraphrase 변형 docs_df (paraphrase_fn 으로 각 문서 재서술).
def build_gpt_variant_df(
    original_df: pd.DataFrame, paraphrase_fn, model: str = "gpt-5.4",
    temperature: float = 0, seed: int = 42,
) -> pd.DataFrame:
    doc_key = f"gpt-para@{model}:s{seed}"
    rows = []
    for row in original_df.itertuples(index=False):
        rows.append({
            "doc_id": build_doc_id(row.family_id, doc_key),
            "dataset": row.dataset,
            "family_id": row.family_id,
            "text": paraphrase_fn(row.text),
            "variant_type": "gpt_paraphrase",
            "variant_level": temperature,
            "variant_seed": seed,
            "source_doc_id": row.doc_id,
            "meta_json": {"model": model, "temperature": temperature, "seed": seed},
        })
    return normalize_docs_df(pd.DataFrame(rows))


# 변형 세트 저장: {dataset}_original_gpt_para_{model_tag}_t{temp}_s{seed}.
def build_and_save_gpt_variant_set(
    original_df: pd.DataFrame, prepared_dir, dataset: str, paraphrase_fn,
    model: str = "gpt-5.4", temperature: float = 0, seed: int = 42,
) -> None:
    df = build_gpt_variant_df(original_df, paraphrase_fn, model, temperature, seed)
    name = f"{dataset}_original_gpt_para_{_model_tag(model)}_t{int(temperature)}_s{seed}"
    save_prepared_set(df, prepared_dir, name)
