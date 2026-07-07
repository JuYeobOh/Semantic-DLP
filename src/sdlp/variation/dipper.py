"""DIPPER (Krishna et al. 2023) paraphrase 변형 생성 — T5-XXL Adversarial tier.

원본 docs_df → DIPPER 가 lexical/order diversity 로 재서술한 변형 docs_df.
- variant_type=dipper, variant_level=lex{L}_order{O}, doc_id={source}__dipper_{level}_s{seed}
- meta {lex,order,sec,in_words,out_words}. protocol 표준 setting (60,60).

재현성 위해 모델 래퍼(DipperParaphraser)까지 이식. 무거운 import(torch/transformers)는 지연 →
config/schema 경로(테스트·품질지표)는 GPU stack 없이 동작. 4bit/8bit 생성은 bitsandbytes 필요:
GPU 머신에서 `uv sync --group dipper` (optional 의존성 그룹, 기본 sync 엔 미포함).
paraphrase_fn(text)->str 주입식 → 테스트는 mock. §8: 기존 dipper parquet 신뢰(재생성 안 함).
⚠️ DIPPER control 은 diversity 의 반대(code=100-diversity). lex=60 → 모델 입력 code=40.
원 구현: https://github.com/martiansideofthemoon/ai-detection-paraphrases
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import nltk
import pandas as pd

from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# DIPPER 학습 격자 (diversity 는 20 단위).
_VALID_DIVERSITY = (0, 20, 40, 60, 80, 100)


# nltk 문장 tokenizer 자원 확보 (nltk>=3.9 는 punkt_tab).
def _ensure_punkt() -> None:
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                nltk.data.find(f"tokenizers/{resource}")
                return
            except Exception:
                continue


# DIPPER control 문자열. code 는 diversity 의 반대(100 - diversity).
def build_control(lex_diversity: int, order_diversity: int) -> str:
    return f"lexical = {100 - lex_diversity}, order = {100 - order_diversity}"


# 텍스트의 토큰 수 (special token 포함).
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


# control + (문장 단위로 왼쪽부터 잘린 prefix) + window 를 budget 토큰 내로 구성.
# 오래된 prefix 문장부터 통째로 drop(문장 중간 절단 없음). prefix 다 버려도 초과하면 그대로 반환.
def fit_input(tokenizer, control: str, prefix_sents: list[str], window_block: str, budget: int) -> tuple[str, int]:
    start = 0
    while True:
        prefix = " ".join(prefix_sents[start:]).strip()
        pieces = [control] + ([prefix] if prefix else []) + [window_block]
        text = " ".join(pieces)
        total = count_tokens(tokenizer, text)
        if total <= budget or start >= len(prefix_sents):
            return text, total
        start += 1


# DIPPER 변형 강도·생성 설정. lex/order 는 0~100 의 20 단위. protocol 표준 (60,60).
@dataclass
class DipperConfig:
    lex_diversity: int = 60
    order_diversity: int = 60
    sent_interval: int = 3
    max_new_tokens: int = 512
    top_p: float = 0.75
    do_sample: bool = True
    max_input_tokens: int = 512   # DIPPER(t5-v1_1-xxl) 선언 입력 길이. 초과 시 prefix 를 문장단위 drop.
    seed: int = 42                # 샘플링(do_sample) 재현용 torch.manual_seed. 원 fullrun 엔 없던 고정.

    # 유효한 diversity 값인지 검증.
    def __post_init__(self) -> None:
        for name, value in (("lex_diversity", self.lex_diversity), ("order_diversity", self.order_diversity)):
            if value not in _VALID_DIVERSITY:
                raise ValueError(f"{name} must be one of {_VALID_DIVERSITY}, got {value}")

    # set 이름·doc_id 에 쓰는 level 태그 (lex60_order60).
    @property
    def level_tag(self) -> str:
        return f"lex{self.lex_diversity}_order{self.order_diversity}"


# DIPPER(T5-XXL 11B) paraphrase 모델 래퍼. torch/transformers 는 생성 시에만 지연 import.
class DipperParaphraser:
    # 모델·토크나이저 로드 (quant: 4bit nf4 기본 | 8bit | none=fp16). 4bit/8bit 는 bitsandbytes 필요.
    def __init__(
        self,
        model_name: str = "kalpeshk2011/dipper-paraphraser-xxl",
        tokenizer_name: str = "google/t5-v1_1-xxl",
        quant: str = "4bit",
        verbose: bool = True,
    ) -> None:
        import torch
        from transformers import BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer

        t0 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        # 양자화 로드는 GPU0 에 강제 배치(accelerate 플래너가 CPU 로 분산하는 경우 방지).
        gpu0 = {"": 0}
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            model_kwargs = {"quantization_config": quant_config, "device_map": gpu0}
        elif quant == "8bit":
            model_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": gpu0}
        elif quant == "none":
            model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        else:
            raise ValueError(f"quant must be '4bit' | '8bit' | 'none', got {quant!r}")

        self.model = T5ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        _ensure_punkt()
        if verbose:
            print(f"[dipper] {model_name} loaded in {time.time() - t0:.1f}s "
                  f"(quant={quant}, device={self.model.device})")

    # 3문장 윈도우를 문서 전체에 슬라이딩하며 paraphrase (prefix 누적, 토큰 budget 내 문장단위 절단).
    def paraphrase(self, input_text: str, cfg: DipperConfig, prefix: str = "") -> str:
        import torch

        with torch.inference_mode():
            torch.manual_seed(cfg.seed)   # 문서별 시드 고정 → 순서·배치 무관 재현
            control = build_control(cfg.lex_diversity, cfg.order_diversity)
            sentences = nltk.sent_tokenize(" ".join(input_text.split()))
            prefix_sents = nltk.sent_tokenize(prefix) if prefix.strip() else []
            output_parts: list[str] = []
            for i in range(0, len(sentences), cfg.sent_interval):
                window = " ".join(sentences[i: i + cfg.sent_interval])
                window_block = f"<sent> {window} </sent>"
                input_str, _ = fit_input(self.tokenizer, control, prefix_sents, window_block, cfg.max_input_tokens)
                encoded = self.tokenizer([input_str], return_tensors="pt", truncation=True,
                                         max_length=cfg.max_input_tokens)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                generated = self.model.generate(**encoded, do_sample=cfg.do_sample, top_p=cfg.top_p,
                                                top_k=None, max_new_tokens=cfg.max_new_tokens)
                text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                output_parts.append(text)
                prefix_sents.extend(nltk.sent_tokenize(text))
            return " ".join(output_parts).strip()

    # 여러 문서를 문서 간 배치로 paraphrase (문서별 독립 prefix/위치 → 문서간 context 누수 없음).
    def paraphrase_batch(self, texts: list[str], cfg: DipperConfig) -> list[str]:
        import torch

        with torch.inference_mode():
            torch.manual_seed(cfg.seed)   # 배치 시작 시 시드 고정(재현은 batch_size·순서에 종속)
            control = build_control(cfg.lex_diversity, cfg.order_diversity)
            docs = [{"sents": nltk.sent_tokenize(" ".join(t.split())), "pos": 0, "prefix_sents": [], "out": []}
                    for t in texts]
            while True:
                active = [d for d in docs if d["pos"] < len(d["sents"])]
                if not active:
                    break
                inputs = []
                for d in active:
                    window = " ".join(d["sents"][d["pos"]: d["pos"] + cfg.sent_interval])
                    window_block = f"<sent> {window} </sent>"
                    input_str, _ = fit_input(self.tokenizer, control, d["prefix_sents"], window_block, cfg.max_input_tokens)
                    inputs.append(input_str)
                encoded = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                         max_length=cfg.max_input_tokens)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                generated = self.model.generate(**encoded, do_sample=cfg.do_sample, top_p=cfg.top_p,
                                                top_k=None, max_new_tokens=cfg.max_new_tokens)
                outs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                for d, out in zip(active, outs):
                    d["out"].append(out)
                    d["prefix_sents"].extend(nltk.sent_tokenize(out))
                    d["pos"] += cfg.sent_interval
            return [" ".join(d["out"]).strip() for d in docs]


# 원본 docs_df → DIPPER 변형 docs_df (paraphrase_fn 으로 각 문서 재서술, sec 계측 포함).
def build_dipper_variant_df(
    original_df: pd.DataFrame, paraphrase_fn, lex: int = 60, order: int = 60, seed: int = 42,
) -> pd.DataFrame:
    level = f"lex{lex}_order{order}"
    rows = []
    for row in original_df.itertuples(index=False):
        source_text = str(row.text)
        t0 = time.perf_counter()
        para = paraphrase_fn(source_text)
        sec = round(time.perf_counter() - t0, 1)
        rows.append({
            "doc_id": f"{row.doc_id}__dipper_{level}_s{seed}",
            "dataset": row.dataset,
            "family_id": row.family_id,
            "text": para,
            "variant_type": "dipper",
            "variant_level": level,
            "variant_seed": seed,
            "source_doc_id": row.doc_id,
            "meta_json": {"lex": lex, "order": order, "sec": sec,
                          "in_words": len(source_text.split()), "out_words": len(para.split())},
        })
    return normalize_docs_df(pd.DataFrame(rows))


# 변형 세트 저장: {dataset}_original_dipper_lex{L}_order{O}_s{seed}.
def build_and_save_dipper_variant_set(
    original_df: pd.DataFrame, prepared_dir, dataset: str, paraphrase_fn,
    lex: int = 60, order: int = 60, seed: int = 42,
) -> None:
    df = build_dipper_variant_df(original_df, paraphrase_fn, lex, order, seed)
    name = f"{dataset}_original_dipper_lex{lex}_order{order}_s{seed}"
    save_prepared_set(df, prepared_dir, name)
