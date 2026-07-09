"""임베딩 설정 (EmbedSpec) — 모델·배치·정규화·device. slug() 는 캐시 키."""
from __future__ import annotations

from dataclasses import dataclass


# 모델 이름 → 경로 안전 slug (캐시 디렉터리 이름). EmbedSpec·longctx 공용.
def model_slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_").replace("@", "_")


# 임베딩 모델과 실행 설정. device=None 은 자동(CUDA 필수), "cpu" 는 경고.
@dataclass(frozen=True)
class EmbedSpec:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None

    # 캐시 디렉터리 이름으로 쓰는 모델 식별 slug.
    def slug(self) -> str:
        return model_slug(self.model_name)
