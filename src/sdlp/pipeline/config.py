"""실행 설정(RunConfig) 과 데이터셋별 default.

ponytail: 청크 인덱스를 공유하는 method 는 여기 method 필드로 분기(chunk_voting/chunk_maxsim).
          검색 메커니즘이 다른 라이벌(longctx/bm25/ssdeep 등)은 methods/ 에 별도 votes 함수.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sdlp.chunking.core import ChunkSpec
from sdlp.embedding.spec import EmbedSpec
from sdlp.ids import sanitize_piece
from sdlp.index.faiss_hnsw import FAISSHNSWConfig

# 청크 인덱스를 공유하는 method (검색 동일, 집계만 다름).
CHUNK_INDEX_METHODS: tuple[str, ...] = ("chunk_voting", "chunk_maxsim")

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
        "original_set": "par3_gt_split",
        "variant_sets": ["par3_human_t1_split"],
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

    method: str = "chunk_voting"                # chunk_voting(투표) | chunk_maxsim(최대 유사도). 검색 동일, 집계만 다름
    top_k: int = 1
    use_score_weight: bool = False              # 논문 default = 표 개수 기반 (chunk_voting 전용)
    include_original_as_positive: bool = True   # 기본 ON: 기밀 원본도 positive(verbatim 유출 포함). OFF=변형만
    save_retrieval: bool = True                 # retrieval_topk.parquet 저장 여부

    # dataset / method 유효성 검사.
    def __post_init__(self) -> None:
        if self.dataset not in DATASET_DEFAULTS:
            raise ValueError(f"모르는 dataset {self.dataset!r}. 선택: {list(DATASET_DEFAULTS)}")
        if self.method not in CHUNK_INDEX_METHODS:
            raise ValueError(f"모르는 method {self.method!r}. 선택: {list(CHUNK_INDEX_METHODS)}")

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

    # 변형 조합 지문 — override 로 돌릴 때 run_dir 충돌 방지 (default 조합은 태그 없음, 기존 경로 유지).
    def _variant_slug(self) -> str:
        resolved = tuple(self.resolved_variant_sets)
        if resolved == tuple(DATASET_DEFAULTS[self.dataset]["variant_sets"]):
            return ""
        digest = hashlib.sha1("|".join(resolved).encode("utf-8")).hexdigest()[:8]
        return f"__var{digest}"

    # 실험 정체성 — runs/<run_ident>/ 의 첫 단계. 검색 방법 + 기밀 DB(original_set) + 쿼리 구성.
    # method: 명시 안 하면 self.method(chunk_voting/chunk_maxsim). 라이벌은 "longctx" 등 직접 전달.
    def run_ident(self, method: str | None = None) -> str:
        ident = f"{method or self.method}__{self.resolved_original_set}__s{self.split_seed}"
        ident += self._variant_slug()
        if self.include_original_as_positive:
            ident += "__inclorig"
        return sanitize_piece(ident)

    # 세부 파라미터 슬러그 — runs/<run_ident>/<config_slug>/ 의 leaf. chunk 계열 파라미터(임베딩·청킹·인덱스·투표).
    # threshold 는 항상 best-F1 로 판별하므로 slug 에 안 넣는다.
    def config_slug(self) -> str:
        vote_tag = "sw" if self.use_score_weight else "cb"
        raw = (
            f"{self.embed_spec.slug()}__{self.chunk_spec.slug()}__"
            f"{self.faiss_config.slug()}__top{self.top_k}__{vote_tag}"
        )
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
            "run_ident": self.run_ident(),
            "config_slug": self.config_slug(),
        }
        return d
