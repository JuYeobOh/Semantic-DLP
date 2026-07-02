"""FAISS HNSW 인덱스 (코사인 = L2 정규화 임베딩 + 내적) — 빌드/저장/로드/검색.

인덱스 생성은 등록(사전 구축) 단계 → 빌드 시간은 추론 latency 에 포함하지 않는다(계측만).
ponytail: 임베딩이 L2 정규화라 내적=코사인 → metric 은 내적 고정 (l2 분기 불필요).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from sdlp.io import ensure_dir, save_json


# HNSW 하이퍼파라미터 (M=노드당 이웃 수, efConstruction=빌드 탐색폭, efSearch=검색 탐색폭).
@dataclass(frozen=True)
class FAISSHNSWConfig:
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    # 캐시/식별용 slug.
    def slug(self) -> str:
        return f"hnsw_M{self.M}_efc{self.ef_construction}_efs{self.ef_search}"


# 빌드된 인덱스 + 청크 메타 + 설정을 묶은 아티팩트.
class FAISSHNSWArtifact:
    def __init__(self, index, meta_df: pd.DataFrame, config: FAISSHNSWConfig) -> None:
        self.index = index
        self.meta_df = meta_df.reset_index(drop=True)
        self.config = config

    # 쿼리 임베딩(N×D)에 대해 top_k 검색 → (scores, indices) 둘 다 N×top_k.
    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(queries.astype(np.float32), top_k)

    # 인덱스·메타·설정을 out_dir 에 저장.
    def save(self, out_dir: str | Path) -> Path:
        out_dir = ensure_dir(out_dir)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        self.meta_df.to_parquet(out_dir / "meta.parquet", index=False)
        save_json(asdict(self.config), out_dir / "config.json")
        return out_dir

    # 저장된 인덱스 로드 (efSearch 복원).
    @classmethod
    def load(cls, out_dir: str | Path) -> "FAISSHNSWArtifact":
        out_dir = Path(out_dir)
        with open(out_dir / "config.json", encoding="utf-8") as f:
            cfg = FAISSHNSWConfig(**json.load(f))
        index = faiss.read_index(str(out_dir / "index.faiss"))
        index.hnsw.efSearch = cfg.ef_search
        meta_df = pd.read_parquet(out_dir / "meta.parquet")
        return cls(index=index, meta_df=meta_df, config=cfg)


# 청크 임베딩으로 HNSW 인덱스 구축 (내적=코사인, 임베딩 L2 정규화 전제).
def build_faiss_hnsw_index(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    config: FAISSHNSWConfig | None = None,
) -> FAISSHNSWArtifact:
    config = config or FAISSHNSWConfig()
    if embeddings.ndim != 2:
        raise ValueError("embeddings 는 2D 여야 함")
    if len(meta_df) != embeddings.shape[0]:
        raise ValueError("meta_df 행 수와 embeddings 행 수가 일치해야 함")

    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], config.M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.ef_construction
    index.hnsw.efSearch = config.ef_search
    index.add(embeddings)
    return FAISSHNSWArtifact(index=index, meta_df=meta_df, config=config)
