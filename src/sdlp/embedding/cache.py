"""임베딩 캐시 — 청크 임베딩 + 메타 + 빌드 소요시간을 디스크에 저장/로드.

경로: artifacts/embeddings/{embed_slug}/{chunk_slug}/{set_name}/
파일: embeddings.npy (N×D float32), meta.parquet (청크 메타), config.json (시간·개수)
config.json 의 embed_sec 덕분에 캐시 적중 시에도 throughput 재현이 가능하다.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sdlp.embedding.spec import EmbedSpec
from sdlp.io import ensure_dir, load_json, save_json
from sdlp.schemas import CHUNK_META_COLUMNS


# (모델 + 청킹설정 + 데이터셋) 을 키로 하는 캐시 디렉터리 경로.
def embedding_cache_dir(
    artifacts_dir: str | Path, embed_spec: EmbedSpec, chunk_slug: str, set_name: str
) -> Path:
    return Path(artifacts_dir) / "embeddings" / embed_spec.slug() / chunk_slug / set_name


# 임베딩·메타·설정(시간 포함) 을 out_dir 에 저장.
def save_embedding_artifact(
    out_dir: str | Path,
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    embed_spec: EmbedSpec,
    chunk_sec: float,
    embed_sec: float,
    n_docs: int,
) -> Path:
    out_dir = ensure_dir(out_dir)
    np.save(out_dir / "embeddings.npy", embeddings)
    chunks_df[CHUNK_META_COLUMNS].to_parquet(out_dir / "meta.parquet", index=False)
    save_json(
        {
            "embed_spec": asdict(embed_spec),
            "n_vectors": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "n_docs": int(n_docs),
            "chunk_sec": float(chunk_sec),
            "embed_sec": float(embed_sec),
        },
        out_dir / "config.json",
    )
    return out_dir


# 캐시에서 (청크 메타, 임베딩) 로드.
def load_embedding_artifact(cache_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    cache_dir = Path(cache_dir)
    embeddings = np.load(cache_dir / "embeddings.npy")
    chunks_df = pd.read_parquet(cache_dir / "meta.parquet")
    return chunks_df, embeddings


# 캐시 config.json (빌드 시간·개수) 로드.
def load_embedding_config(cache_dir: str | Path) -> dict:
    return load_json(Path(cache_dir) / "config.json")


# 캐시가 존재(임베딩+메타 파일 모두)하는지 확인.
def embedding_cache_exists(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    return (cache_dir / "embeddings.npy").exists() and (cache_dir / "meta.parquet").exists()
