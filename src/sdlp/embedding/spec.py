"""임베딩 설정 (EmbedSpec) — 모델·배치·정규화·device. slug() 는 캐시 키."""
from __future__ import annotations

from dataclasses import dataclass


# 임베딩 모델과 실행 설정. device=None 은 자동(CUDA 필수), "cpu" 는 경고.
@dataclass(frozen=True)
class EmbedSpec:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None

    # 캐시 디렉터리 이름으로 쓰는 모델 식별 slug (경로 안전 문자로 치환).
    def slug(self) -> str:
        return self.model_name.replace("/", "__").replace(":", "_").replace("@", "_")
