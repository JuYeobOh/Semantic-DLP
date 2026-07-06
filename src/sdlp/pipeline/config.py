"""실행 설정(RunConfig) 과 데이터셋별 default.

ponytail: 지금은 embedding(chunk_voting) method 하나. 라이벌(ssdeep/bm25/longctx)은 S11 에서
          method 필드 + 분기 추가.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from sdlp.chunking.core import ChunkSpec
from sdlp.embedding.spec import EmbedSpec
from sdlp.ids import sanitize_piece
from sdlp.index.faiss_hnsw import FAISSHNSWConfig

# 데이터셋별 기본 original_set / variant_sets (None 이면 이걸 사용).
DATASET_DEFAULTS: dict[str, dict] = {
    "semeval": {
        "original_set": "semeval_original",
        "variant_sets": [
            "semeval_original_eda_sr_a10_s42",
            "semeval_original_gpt_para_54_t0_s42",
            "semeval_original_dipper_lex60_order60_s42",
        ],
    },
    "krapivin": {
        "original_set": "krapivin_original",
        "variant_sets": [
            "krapivin_original_eda_sr_a10_s42",
            "krapivin_original_gpt_para_54_t0_s42",
            "krapivin_original_dipper_lex60_order60_s42",
        ],
    },
    "re3sci": {
        "original_set": "re3_v1_only",
        "variant_sets": ["re3_v2_only"],
    },
    "casimir": {   # dedup 세트는 S9 에서 재생성
        "original_set": "casimir_v1_only_dedup",
        "variant_sets": ["casimir_latest_only_dedup"],
    },
    "par3": {      # 기밀 DB=GoogleMT 전문(분권 병합). 기본 positive=사람 번역가 1번(translator_1)만.
        # override: "사람 전부"=variant_sets=("par3_human",) / 분권 분리=par3_gt_split·par3_human_t1_split
        "original_set": "par3_gt",
        "variant_sets": ["par3_human_t1"],
    },
}


# 한 실행의 모든 설정 (frozen — 캐시/slug 키로 안전).
@dataclass(frozen=True)
class RunConfig:
    dataset: str

    prepared_dir: str | Path = "data/prepared"
    splits_dir: str | Path = "data/splits"
    artifacts_dir: str | Path = "artifacts"

    original_set: str | None = None
    variant_sets: tuple[str, ...] | None = None   # tuple = frozen 친화

    split_seed: int = 42
    split_ratio: float = 0.5

    chunk_spec: ChunkSpec = field(default_factory=ChunkSpec)
    embed_spec: EmbedSpec = field(default_factory=EmbedSpec)
    faiss_config: FAISSHNSWConfig = field(default_factory=FAISSHNSWConfig)

    top_k: int = 1
    confidence_threshold: float = 0.5
    use_score_weight: bool = False              # 논문 default = 표 개수 기반
    include_original_as_positive: bool = False  # ON 이면 기밀 원본도 positive
    save_retrieval: bool = True                 # retrieval_topk.parquet 저장 여부

    # dataset 유효성 검사.
    def __post_init__(self) -> None:
        if self.dataset not in DATASET_DEFAULTS:
            raise ValueError(f"모르는 dataset {self.dataset!r}. 선택: {list(DATASET_DEFAULTS)}")

    # 해석된 original_set (None 이면 default).
    @property
    def resolved_original_set(self) -> str:
        return self.original_set or DATASET_DEFAULTS[self.dataset]["original_set"]

    # 해석된 variant_sets (None 이면 default).
    @property
    def resolved_variant_sets(self) -> list[str]:
        if self.variant_sets is not None:
            return list(self.variant_sets)
        return list(DATASET_DEFAULTS[self.dataset]["variant_sets"])

    # 실행 산출물 디렉터리 이름 (결과에 영향 주는 knob 들을 인코딩).
    def run_slug(self) -> str:
        vote_tag = "sw" if self.use_score_weight else "cb"   # sw=score-weighted, cb=count based
        thr = int(round(self.confidence_threshold * 100))
        raw = (
            f"{self.dataset}__s{self.split_seed}__{self.chunk_spec.slug()}__"
            f"{self.embed_spec.slug()}__{self.faiss_config.slug()}__"
            f"top{self.top_k}__t{thr}__{vote_tag}"
        )
        if self.include_original_as_positive:
            raw += "__inclorig"
        return sanitize_piece(raw)

    # yaml/json 저장용 dict (해석된 값 + slug 포함).
    def as_serializable(self) -> dict:
        d = asdict(self)
        d["prepared_dir"] = str(self.prepared_dir)
        d["splits_dir"] = str(self.splits_dir)
        d["artifacts_dir"] = str(self.artifacts_dir)
        if d.get("variant_sets") is not None:
            d["variant_sets"] = list(d["variant_sets"])
        d["_resolved"] = {
            "original_set": self.resolved_original_set,
            "variant_sets": self.resolved_variant_sets,
            "run_slug": self.run_slug(),
        }
        return d
