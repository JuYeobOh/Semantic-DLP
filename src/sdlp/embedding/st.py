"""SentenceTransformer 기반 텍스트 임베더 (GPU 전제).

device 정책: 자동(None)/cuda 요청 시 CUDA 없으면 에러, 명시적 cpu 는 경고.
"""
from __future__ import annotations

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from sdlp.embedding.spec import EmbedSpec


# 요청 device 를 검증·해석. 실험은 GPU 전제 → CUDA 없으면 에러, cpu 명시는 경고.
def resolve_device(device: str | None) -> str:
    if device == "cpu":
        warnings.warn("CPU 로 임베딩 — 매우 느림. 논문 실험은 GPU 에서 실행하세요.")
        return "cpu"
    if device is None or device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU 가 필요합니다 (torch.cuda.is_available()=False). "
                "GPU 환경에서 실행하거나 device='cpu' 로 명시하세요."
            )
        return device or "cuda"
    return device


# SentenceTransformer 를 감싼 임베더. chunks_df 를 임베딩 + 소요시간 측정.
class STTextEmbedder:
    # 모델 로드 (device 정책 적용).
    def __init__(self, spec: EmbedSpec) -> None:
        self.spec = spec
        self.model = SentenceTransformer(spec.model_name, device=resolve_device(spec.device))

    # 청킹 token 모드에서 쓰는 모델 tokenizer 접근자.
    @property
    def tokenizer(self):
        module = self.model._first_module()
        if hasattr(module, "tokenizer"):
            return module.tokenizer
        raise AttributeError("SentenceTransformer 안에서 tokenizer 를 찾지 못함.")

    # 텍스트 리스트를 임베딩 (N×D float32). 빈 리스트는 빈 배열.
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return vecs.astype(np.float32)

    # chunks_df['chunk_text'] 를 임베딩. 반환: (임베딩, 소요초).
    def encode_chunks_df(self, chunks_df: pd.DataFrame) -> tuple[np.ndarray, float]:
        start = perf_counter()
        embeddings = self.encode_texts(chunks_df["chunk_text"].tolist())
        return embeddings, perf_counter() - start
