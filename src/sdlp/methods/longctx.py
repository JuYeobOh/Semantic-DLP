"""embedding_longctx — 문서 통짜(no chunking) long-context 임베딩 top-1 매칭.

각 문서를 single 벡터로 인코딩 → query×ref cosine → 문서/family top-1.
**confidence = best-family cosine**(raw; threshold sweep·AUC 로 평가). 이름을 confidence 로 둬야
chunk_voting 과 동일한 evaluate_run/apply_threshold 경로에 drop-in 된다(값은 cosine).
기본 모델: ibm-granite/granite-embedding-97m-multilingual-r2 (encoder bi-encoder, context 32768, dim 384, 경량).
성능형 대안: ibm-granite/granite-embedding-311m-multilingual-r2 (dim 768, Matryoshka).

max_seq_length 초과 문서 처리(long_doc):
  "truncate"  = head 절단(앞 N 토큰 보존, 뒤 유실) — 기존 방식
  "mean_pool" = max_seq_length 윈도우로 쪼개 각각 인코딩 후 평균+정규화 (전문 보존)
  "exclude"   = 초과 문서 제외 (실험에서 빠짐)

임베딩 캐시는 **prepared set 단위**(artifacts/cache/embeddings_longctx/{model}/{set}__{long_doc}...npy) —
키가 split·변형조합·inclorig 와 무관하므로 변형별 run 들이 같은 set 캐시를 공유한다(섞임 원천 차단).
실행 시 각 set 전체를 인코딩(캐시)한 뒤 ref/query doc_id 로 골라 조립한다. encode_sec 은 사용 문서수
비례 배분(§4 참고용). 구(뭉텅이 query) 캐시는 split_combined_query_cache 로 분리(재인코딩 0).
torch/sentence_transformers 는 지연 import. votes_df 는 FINAL VOTES 스키마(confidence) 호환.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from sdlp.embedding.spec import model_slug

DEFAULT_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
DEFAULT_MAX_SEQ = 32768   # granite r2 context (97m/311m 모두 32k)
LONG_DOC_MODES = ("truncate", "mean_pool", "exclude")

# 프로세스 내 모델 재사용 (같은 설정이면 1회 로드).
_MODEL_CACHE: dict[str, object] = {}


# SentenceTransformer 로드 (지연 import, 프로세스 캐시). device None → cuda 있으면 cuda.
def _get_encoder(model_name: str, max_seq_length: int, device: str | None, dtype: str):
    import torch
    from sentence_transformers import SentenceTransformer

    key = f"{model_name}|{device}|{dtype}|{max_seq_length}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 초장문(32k) attention 을 math 커널(N×N 행렬 materialize)로 돌리면 OOM →
    # mem-efficient/flash SDPA 강제, math 끔. (짧은 문서엔 영향 없음.)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"torch_dtype": torch_dtype}, trust_remote_code=True)
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.truncation_side = "right"   # head truncation(앞 N 토큰 보존)
    model.max_seq_length = int(max_seq_length)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


# 문서 텍스트를 doc-level 벡터로 인코딩 (long_doc 전략 적용).
# 반환: (emb(N,D), 유지된 doc_ids, family_ids, n_overflow). exclude 는 유지 문서만, 나머지는 전체.
def _encode_docs(model, texts, doc_ids, family_ids, max_seq_length, batch_size, long_doc):
    def _enc(seg_texts, normalize):
        if not seg_texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), np.float32)
        return model.encode(seg_texts, batch_size=batch_size, normalize_embeddings=normalize,
                            convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    tok = model.tokenizer
    ids_list = tok(texts, truncation=False, padding=False, add_special_tokens=True,
                   return_attention_mask=False)["input_ids"]
    lengths = [len(x) for x in ids_list]
    n_overflow = int(sum(l > max_seq_length for l in lengths))

    if long_doc == "exclude":
        keep = [i for i, l in enumerate(lengths) if l <= max_seq_length]
        emb = _enc([texts[i] for i in keep], normalize=True)
        return emb, [doc_ids[i] for i in keep], [family_ids[i] for i in keep], n_overflow

    if long_doc == "mean_pool":
        win = max(1, max_seq_length - 2)   # 특수토큰 여유
        seg_texts, seg_owner = [], []
        for di, ids in enumerate(ids_list):
            if len(ids) <= max_seq_length:
                seg_texts.append(texts[di]); seg_owner.append(di)
            else:
                body = ids[1:-1] if len(ids) >= 2 else ids
                for s in range(0, len(body), win):
                    seg_texts.append(tok.decode(body[s: s + win], skip_special_tokens=True))
                    seg_owner.append(di)
        seg_emb = _enc(seg_texts, normalize=False)   # 평균 전엔 정규화 X
        out = np.zeros((len(texts), seg_emb.shape[1]), np.float32)
        cnt = np.zeros(len(texts))
        for e, owner in zip(seg_emb, seg_owner):
            out[owner] += e; cnt[owner] += 1
        cnt[cnt == 0] = 1
        out /= cnt[:, None]
        out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)   # 세그먼트 평균 후 L2 정규화
        return out, list(doc_ids), list(family_ids), n_overflow

    # truncate (기본): SBERT 가 max_seq_length 로 head 절단
    return _enc(texts, normalize=True), list(doc_ids), list(family_ids), n_overflow


# 한 prepared-set **전체**를 doc-level 벡터로 인코딩 (set 단위 npy 캐시 — split·변형·inclorig 무관).
# 반환: (emb, meta[doc_id,family_id], timing{encode_sec,source,n_overflow}). 미스 시에만 set 로드+인코딩.
def _encode_set(set_name, model_name, max_seq_length, batch_size, device, dtype,
                artifacts_dir, prepared_dir, long_doc="truncate", force_rebuild=False):
    cache_root = Path(artifacts_dir) / "cache" / "embeddings_longctx" / model_slug(model_name)
    cache_root.mkdir(parents=True, exist_ok=True)
    # 키에 max_seq_length·dtype 포함 — 설정 바꾸면 새 캐시로 갈려 stale 재사용 방지.
    tag = f"{set_name}__{long_doc}__L{max_seq_length}__{dtype}"
    npy_p = cache_root / f"{tag}.npy"
    meta_p = cache_root / f"{tag}.parquet"
    info_p = cache_root / f"{tag}.meta.json"

    if not force_rebuild and npy_p.exists() and meta_p.exists() and info_p.exists():
        emb = np.load(npy_p)
        meta_df = pd.read_parquet(meta_p)
        info = json.loads(info_p.read_text(encoding="utf-8"))
        return emb, meta_df, {"encode_sec": float(info.get("encode_sec", 0.0)), "source": "cached",
                              "n_overflow": int(info.get("n_overflow", 0))}

    from sdlp.io import load_prepared_set
    docs_df = load_prepared_set(prepared_dir, set_name)
    model = _get_encoder(model_name, max_seq_length, device, dtype)
    sub = docs_df[["doc_id", "family_id", "text"]].sort_values("doc_id").reset_index(drop=True)
    texts = [t if str(t).strip() else " " for t in sub["text"].fillna("").astype(str)]

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # 이전 인코딩의 예약메모리 반환 → 파편화로 인한 OOM 완화

    t0 = perf_counter()
    emb, kept_doc, kept_fam, n_overflow = _encode_docs(
        model, texts, sub["doc_id"].tolist(), sub["family_id"].tolist(),
        max_seq_length, batch_size, long_doc)
    encode_sec = perf_counter() - t0

    meta_df = pd.DataFrame({"doc_id": kept_doc, "family_id": kept_fam})
    np.save(npy_p, emb)
    meta_df.to_parquet(meta_p, index=False)
    info_p.write_text(json.dumps({"model_name": model_name, "max_seq_length": int(max_seq_length),
                                  "long_doc": long_doc, "encode_sec": float(encode_sec),
                                  "n_docs": int(len(meta_df)), "n_overflow": int(n_overflow)},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return emb, meta_df, {"encode_sec": float(encode_sec), "source": "live", "n_overflow": int(n_overflow)}


# set 별 encode_sec 을 "사용한 문서수 / set 전체 문서수" 비례로 배분 (§4 참고용 latency).
def _prorated_encode_sec(per_set: list[tuple[set, float]], used_ids: set) -> float:
    total = 0.0
    for ids, sec in per_set:
        if ids:
            total += sec * (len(used_ids & ids) / len(ids))
    return total


# rival 계약: (reference, query) → (votes_df, timing). 문서 통짜 임베딩을 기밀 풀(FAISS)에 등록 → 벡터서치.
# set_names: 이 run 을 구성하는 prepared set 들(원본 + 변형들) — 각각 set 단위로 인코딩(캐시) 후
# ref/query doc_id 로 골라 조립한다. long_doc: 초과 문서 처리 전략 ("truncate" | "mean_pool" | "exclude").
def longctx_votes(
    reference_docs_df: pd.DataFrame,
    query_docs_df: pd.DataFrame,
    set_names: list[str],
    prepared_dir: str | Path = "data/prepared",
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ,
    long_doc: str = "truncate",
    batch_size: int = 32,
    device: str | None = None,
    dtype: str = "float32",
    artifacts_dir: str | Path = "artifacts",
    faiss_config=None,
    top_k: int = 50,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if long_doc not in LONG_DOC_MODES:
        raise ValueError(f"long_doc must be one of {LONG_DOC_MODES}, got {long_doc!r}")
    if faiss_config is None:
        from sdlp.index.faiss_hnsw import FAISSHNSWConfig
        faiss_config = FAISSHNSWConfig()

    t0 = perf_counter()
    # ---- set 단위 인코딩(캐시) → doc_id → 벡터 ----
    vec: dict[object, np.ndarray] = {}
    per_set: list[tuple[set, float]] = []   # (set 의 doc_id 집합, encode_sec)
    sources: list[str] = []
    n_overflow_total = 0
    for s in set_names:
        emb, meta, t = _encode_set(s, model_name, max_seq_length, batch_size, device, dtype,
                                   artifacts_dir, prepared_dir, long_doc, force_rebuild)
        ids = meta["doc_id"].tolist()
        vec.update(zip(ids, emb))
        per_set.append((set(ids), float(t["encode_sec"])))
        sources.append(t["source"])
        n_overflow_total += int(t["n_overflow"])

    # ---- ref/query 조립 (exclude 모드로 빠진 문서는 제외) ----
    ref_ids = [d for d in reference_docs_df["doc_id"] if d in vec]
    q_ids = [d for d in query_docs_df["doc_id"] if d in vec]
    ref_fam = dict(zip(reference_docs_df["doc_id"], reference_docs_df["family_id"]))
    q_fam = dict(zip(query_docs_df["doc_id"], query_docs_df["family_id"]))
    dim = next(iter(vec.values())).shape[0] if vec else 0
    ref_emb = np.stack([vec[d] for d in ref_ids]) if ref_ids else np.zeros((0, dim), np.float32)
    q_emb = np.stack([vec[d] for d in q_ids]) if q_ids else np.zeros((0, dim), np.float32)

    ref_encode_sec = _prorated_encode_sec(per_set, set(ref_ids))
    query_encode_sec = _prorated_encode_sec(per_set, set(q_ids))

    # 문서벡터 → 기밀 풀(FAISS) 등록 + 벡터서치 + maxsim (chunk_maxsim 과 동일 tail, docvec 공용).
    from sdlp.methods.docvec import doc_vector_maxsim
    votes, index_build_sec, search_sec = doc_vector_maxsim(
        ref_emb, ref_ids, [ref_fam[d] for d in ref_ids],
        q_emb, q_ids, [q_fam[d] for d in q_ids],
        faiss_config, top_k=top_k)

    timing = {
        "long_doc": long_doc,
        # encode_sec 은 set 캐시 기록값을 사용 문서수 비례로 배분(참고용).
        "ref_encode_sec": ref_encode_sec, "query_encode_sec": query_encode_sec,
        "search_sec": search_sec, "index_build_sec": index_build_sec, "total_sec": perf_counter() - t0,
        # §4: inference = 쿼리 임베딩 + 벡터서치. ref 임베딩·인덱스 빌드는 등록이라 제외.
        "inference_total_sec": query_encode_sec + search_sec,
        "build_sec": ref_encode_sec + index_build_sec,
        "embed_source": "cached" if set(sources) == {"cached"} else ("live" if set(sources) == {"live"} else "mixed"),
        "n_overflow_sets": n_overflow_total,   # 구성 set 전체 기준(참고용)
        "model_name": model_name, "max_seq_length": int(max_seq_length),
    }
    return votes, timing


# 구(뭉텅이) query 캐시({orig}__s{seed}[__inclorig]__query 키)를 prepared set 단위 캐시로 분리.
# inclorig 뭉텅이는 원본+기본조합 변형의 전 문서를 포함하므로 행 선택만으로 마이그레이션(재인코딩 0).
# encode_sec·n_overflow 는 문서수 비례 배분(참고용). 커버가 100% 아닌 set 은 건너뜀(신규 인코딩 대상).
def split_combined_query_cache(artifacts_dir, prepared_dir, model_name, combined_base, set_names,
                               long_doc="mean_pool", max_seq_length=DEFAULT_MAX_SEQ,
                               dtype="bfloat16", overwrite=False) -> list[dict]:
    from sdlp.io import load_prepared_set

    root = Path(artifacts_dir) / "cache" / "embeddings_longctx" / model_slug(model_name)
    suffix = f"__{long_doc}__L{max_seq_length}__{dtype}"
    src = root / f"{combined_base}{suffix}"
    emb = np.load(f"{src}.npy")
    meta = pd.read_parquet(f"{src}.parquet")
    info = json.loads(Path(f"{src}.meta.json").read_text(encoding="utf-8"))
    row_of = {d: i for i, d in enumerate(meta["doc_id"])}

    out: list[dict] = []
    for s in set_names:
        dst = root / f"{s}{suffix}"
        if not overwrite and Path(f"{dst}.npy").exists():
            out.append({"set": s, "status": "exists(스킵)"})
            continue
        ids = load_prepared_set(prepared_dir, s)["doc_id"].tolist()
        rows = [row_of[d] for d in ids if d in row_of]
        if len(rows) < len(ids):
            out.append({"set": s, "status": f"커버 {len(rows)}/{len(ids)} — 스킵(신규 인코딩 필요)"})
            continue
        share = len(rows) / max(len(meta), 1)
        np.save(f"{dst}.npy", emb[rows])
        meta.iloc[rows].reset_index(drop=True).to_parquet(f"{dst}.parquet", index=False)
        Path(f"{dst}.meta.json").write_text(json.dumps({
            "model_name": model_name, "max_seq_length": int(max_seq_length), "long_doc": long_doc,
            "encode_sec": float(info.get("encode_sec", 0.0)) * share,   # 문서수 비례 배분(참고용)
            "n_docs": len(rows), "n_overflow": round(float(info.get("n_overflow", 0)) * share),
            "migrated_from": combined_base}, ensure_ascii=False, indent=2), encoding="utf-8")
        out.append({"set": s, "status": f"ok ({len(rows)} docs)"})
    return out
