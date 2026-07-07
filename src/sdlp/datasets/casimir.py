"""CASIMIR 논문 리비전 데이터셋 (HuggingFace taln-ls2n/CASIMIR) → dedup cluster 단위 docs_df.

HF 만으로 전 과정 재현: 본문복원 → forum 신호(서지+문자열) → cross-venue dedup → cluster docs.

핵심 정책:
- 같은 논문이 여러 forum(워크숍+본학회, split 교차)으로 재제출된 것을 하나의 cluster(=family)로 묶는다.
  dedup 신호 = ① prefix(제목+초록 앞) Levenshtein 유사도 ≥ 0.82  ② OpenReview paperhash 정확일치
  ③ 사람이 원문대조로 확정한 41쌍(MANUAL_MERGES). content 유사도는 절대 미사용(라이벌 baseline 순환 방지).
- cluster 대표 = 가장 이른(tcdate 최소) forum = 원본. family = cluster.
- cluster 안 **모든 forum의 모든 version 을 pool** → 제출시각(cdate 우선, 결측 tcdate) 오름차순 → version_index 1..n.
  v1=원본(가장 이른)=기밀 DB, latest=가장 늦은 리비전=positive 쿼리.
- ⚠️ mapping.references 배열 순서는 최신-우선 내림차순이라 version 정렬에 쓰지 않는다. 날짜(metadata)만 신뢰.
  references 는 version→forum 연결 fallback 용으로만 사용.

전체 splits(train+val+test) 15,646 forum → 고유 논문 15,177. 빌드는 HF 대용량이라 사용자가 실행.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sdlp.ids import build_doc_id, build_family_id, make_revision_doc_key
from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# HF split 이름 (전체 사용).
CASIMIR_SPLITS = ("train", "validation", "test")

# ── dedup 상수 ─────────────────────────────────────────────────────────────
# 신호 A 문턱: prefix Levenshtein 유사도. 0.82~0.90 전수검증 결과 오탐 0.
EDIT_TAU = 0.82
# prefix 계산에서 걷어낼 boiler 줄(논문 아닌 공통 머리말). NeurReps/Geometry 워크숍 PDF의 FP 주범 포함.
BOILER = re.compile(
    r"under review|anonymous author|conference paper at|paper under|double-blind|"
    r"affiliationaddress|address email|extended abstract track|^editors:", re.I)

# 사람이 원문 대조로 확정한 version-duplicate 41쌍(자동 A∪B 미포착: 개명/익명↔실명/어순·venue 변경).
# [0.5,0.82) gray-zone 39쌍 → merge 35 / false_positive 4(무병합). + 개명·약어·다forum 6쌍(prefix 블록 밖이라 A로 불가)
# = 41 튜플(distinct cluster 병합 감소분 별도). 옆 주석=논문명. AbLj…는 3-forum 한 논문(2튜플).
MANUAL_MERGES = [
    ("jC9G3ns6jH", "FlrQGoHPcvo"),    # DNN selective inference (statistical significance)
    ("BZO_i51pl5", "-5rFUTO2NWe"),    # Object Representations as Equilibria/Fixed Points
    ("DBBF7G4BMOj", "awOrpNtsCX"),    # Shape-Tailored Deep Neural Networks
    ("oHoAwwQuVO_", "rvsbw2YthH_"),   # contrastive: label efficiency vs universality
    ("Wfcbb0d7UEs", "ZLsZmNe1RDb"),   # How to talk so AI will learn (pragmatics→autonomy)
    ("QuObT9BTWo", "vriLTB2-O0G"),    # Pareto Set Learning (MOCO→MOO/MOBO)
    ("HRF6T1SsyDn", "4FlyRlNSUh"),    # hypergraph NN expressiveness/generalization
    ("BZO_i51pl5", "SSgxDBOIqgq"),    # Object Representations (삼각: 위 두 개와 한 클러스터)
    ("SSgxDBOIqgq", "-5rFUTO2NWe"),   #  "
    ("_gZf4NEuf0H", "wjqr6aqkLUV"),   # Condensation of Neural Networks at Initial Training
    ("ryeNPi0qKX", "BJeYYeaVJ7"),     # Language Modeling Teaches You More Syntax Than Translation
    ("7HPmTa_FdY", "3oWo92cQyxL"),    # multimodal few-shot: meta-mapper
    ("BJluxREKDB", "HkeyZhC9F7"),     # RL for QBF/automated reasoning heuristic
    ("iw-ms2znSS2", "b6to5kfFhQh"),   # Sokoban planning: policy/value, left heavy tails
    ("vRrFVHxFiXJ", "Wz9OtQYk_A"),    # single-cell drug perturbation (chemCPA)
    ("de1kSNxv5BQ", "Rf58LPCwJj0"),   # Optimal Representations for Covariate Shift(s)
    ("uzqUp0GjKDu", "CbxgFfEEP7P"),   # Unsupervised Learning under Latent Label Shift
    ("4u25M570Iji", "4JLiaohIk9"),    # Motion Forecasting with Unlikelihood Training
    ("acShf51GxE", "uyEYNg2HHFQ"),    # Hyper-Representations (→ generative)
    ("i4qKmHdq6y8", "Zo9MZCOn0u"),    # Learning to Abstain (Uninformative Data)
    ("2nJdh_C-UWe", "q3F0UBAruO"),    # MOBA human-AI/agent collaboration
    ("8dB6Hl9HHWF", "dpuLRRQ7zC"),    # SWEEN certified robustness
    ("JyI9lc8WxW", "Pia70sP2Oi1"),    # Planckian Jitter
    ("BR1qoDGxjWp", "EiQB09V5IX"),    # Feed-Forward Latent Domain Adaptation
    ("SRxzjCGMHWc", "SawenqFzFb9"),   # UserIdentifier (익명↔실명)
    ("32Ryt4pAHeD", "yRMehOHpRCy"),   # Explainable RL via Model Transforms
    ("Wo1HF2wWNZb", "BW44SrOU9g5"),   # nonlinear ICA identifiability
    ("IEKL-OihqX0", "9DZKk85Z4zA"),   # gradient-guided importance sampling EBM
    ("56l7PBjbaN", "5KP2cXxnOVx"),    # Hamiltonian Policy Optimization
    ("TTeMp6953v4", "TFbwV6I0VLg"),   # SlotFormer
    ("hq7vLjZTJPk", "uLhKRH-ovde"),   # communication-efficient distributed gradient clipping
    ("WRmTnEOk0E", "E4EE_ohFGz"),     # Diurnal or Nocturnal (periodic-shift FL)
    ("BW5PuV4V-rL", "HyeKcgHFvS"),    # gradient-based training of GMMs
    ("iuSDDiqacPj", "MTex8qKavoS"),   # MetaShift dataset
    ("yzDTTtlIlMr", "i-8uqlurj1f"),   # momentum implicit bias on separable data
    ("CFxHg2L902W","TGUp8EaCGj9"),  # Offline RL at Multiple Frequencies (약어 RL↔Reinforcement Learning)
    ("B1G9doA9F7","HJxjSR5so7"),    # Augmented Cyclic Adversarial Learning for DA (익명↔실명+개명)
    ("aqpOCAlY9Tn","zH9GcZ3ZGXu"),  # Invertible Output Mapping→Feature Reconstruction (개명)
    ("YmONQIWli--","gEoVDSASC2h"),  # Gotta Go Fast (Score-Based Generative Models)
    ("AbLj0l8YbYt","o8_QHMYOfu"),   # Grounding Aleatoric Uncertainty in UED
    ("AbLj0l8YbYt","wYqLTy4wkor"),  #  " (AbLj… 축으로 3-forum 한 논문)
]


# ── HF 로더 (전체 splits 병합) ─────────────────────────────────────────────

# article_pairs config 의 지정 split 들을 하나의 DataFrame 으로 (문장쌍 → 본문 복원용).
def load_article_pairs_df(splits: tuple[str, ...] = CASIMIR_SPLITS) -> pd.DataFrame:
    from datasets import load_dataset  # ponytail: 지연 import — 순수 로직 단위테스트는 datasets 불필요
    ds = load_dataset("taln-ls2n/CASIMIR", "article_pairs")
    return pd.concat([ds[s].to_pandas() for s in splits], ignore_index=True)


# mapping config 의 지정 split 병합 (split 컬럼 부여; id_forum → references[version_id]).
def load_mapping_df(splits: tuple[str, ...] = CASIMIR_SPLITS) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("taln-ls2n/CASIMIR", "mapping")
    return pd.concat([ds[s].to_pandas().assign(split=s) for s in splits], ignore_index=True)


# metadata config (split 없음, 전체 공유). id=version_id, forum, cdate/tcdate, content(JSON: paperhash 등).
def load_metadata_df() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("taln-ls2n/CASIMIR", "metadata")
    return ds["train"].to_pandas()


# ── 파싱 헬퍼 ──────────────────────────────────────────────────────────────

# 결측(None/NaN) 판정. numpy 배열 등 애매값은 결측 아님으로 처리.
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


# 공백·비가시문자(nbsp/CR/개행)를 하나의 space 로 정규화한 단일 문자열.
def _clean_text(value: object) -> str:
    if _is_missing(value):
        return ""
    # split() 은 nbsp( )·CR·개행 등 모든 유니코드 공백을 하나의 space 로 정규화.
    return " ".join(str(value).split()).strip()


# list/np.ndarray/JSON·python 리스트 문자열 → 문자열 리스트.
# np.ndarray 처리가 정본 forum 그룹핑의 핵심 (numpy 배열 미파싱 → references 그룹핑 실패).
def _parse_listlike(value: object) -> list[str]:
    if isinstance(value, (list, np.ndarray)):
        return [_clean_text(x) for x in value if _clean_text(x)]
    if _is_missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
                if isinstance(parsed, list):
                    return [_clean_text(x) for x in parsed if _clean_text(x)]
            except Exception:
                pass
    return []


# content(JSON 문자열 또는 dict)에서 OpenReview paperhash 추출 (없으면 "").
def _paperhash(content: object) -> str:
    if isinstance(content, dict):
        data = content
    else:
        try:
            data = json.loads(content) if isinstance(content, str) else {}
        except Exception:
            data = {}
    return _clean_text(data.get("paperhash")) if isinstance(data, dict) else ""


# 문장 타입을 title/abstract/p 로 정규화 (그 외는 원문 소문자 그대로).
_SENTENCE_TYPE = {
    "title": "title", "article-title": "title",
    "abstract": "abstract", "paragraph": "p", "p": "p",
}


def _canonical_sentence_type(value: object) -> str:
    key = _clean_text(value).lower()
    return _SENTENCE_TYPE.get(key, key)


# 문장 중복 제거용 안정 해시 (sentence_id 결측 시 surrogate key).
def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# 컬럼이 없을 수 있는 article_pairs 에서 안전하게 시리즈 추출.
def _col(df: pd.DataFrame, name: str, default: object) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


# ── 버전 텍스트 복원 (article_pairs 문장쌍 → version 별 본문) ────────────────

# article_pairs 한쪽(side 1|2) 문장들을 version 단위 문장표로 펼침 (title/abstract/p 만 유지).
def _build_side_sentence_frame(pairs_df: pd.DataFrame, side: int) -> pd.DataFrame:
    sub = pd.DataFrame({
        "version_id": _col(pairs_df, f"id_version_{side}", "").map(_clean_text),
        "sentence_id": _col(pairs_df, f"id-sentence-{side}", "").map(_clean_text),
        "text": _col(pairs_df, f"text-sentence-{side}", "").map(_clean_text),
        "sentence_type": _col(pairs_df, f"type-sentence-{side}", "").map(_canonical_sentence_type),
        "page": _col(pairs_df, f"page-sentence-{side}", pd.NA),
        "num_section": _col(pairs_df, f"num_section-sentence-{side}", pd.NA),
        "num_paragraph": _col(pairs_df, f"num_paragraph-sentence-{side}", pd.NA),
        "num_sentence": _col(pairs_df, f"num_sentence-sentence-{side}", pd.NA),
        "pair_index": _col(pairs_df, "sentence-pair-index", pd.NA),
        "side": side,
    })
    sub = sub[
        (sub["version_id"] != "")
        & (sub["text"] != "")
        & (sub["sentence_type"].isin({"title", "abstract", "p"}))
    ].copy()
    # "Abstract" 헤더만 단독으로 있는 줄 제거 (복원 시 우리가 헤더를 다시 붙임).
    sub = sub[~((sub["sentence_type"] == "abstract") & (sub["text"].str.lower() == "abstract"))].copy()

    # sentence_id 결측 → 위치+내용 기반 surrogate key.
    missing = sub["sentence_id"] == ""
    sub.loc[missing, "sentence_id"] = sub[missing].apply(
        lambda r: f"{r['version_id']}::{r['page']}::{r['num_section']}::"
                  f"{r['num_paragraph']}::{r['num_sentence']}::{_stable_hash(r['text'])}",
        axis=1,
    )
    for col in ("page", "num_section", "num_paragraph", "num_sentence", "pair_index"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(10 ** 15)
    return sub


# 양쪽 side 문장을 모아 (version_id, sentence_id) 기준 중복 제거한 version 문장표.
def build_version_sentence_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    sent = pd.concat(
        [_build_side_sentence_frame(pairs_df, 1), _build_side_sentence_frame(pairs_df, 2)],
        ignore_index=True,
    )
    sort_cols = ["version_id", "page", "num_section", "num_paragraph",
                 "num_sentence", "pair_index", "side", "sentence_id"]
    sent = sent.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return sent.drop_duplicates(subset=["version_id", "sentence_id"], keep="first").reset_index(drop=True)


# 한 version 의 문장표를 위치순으로 이어붙여 본문 텍스트로 (abstract 는 'Abstract' 헤더 부여).
def _compose_version_text(group: pd.DataFrame) -> str:
    group = group.sort_values(
        ["page", "num_section", "num_paragraph", "num_sentence", "pair_index", "side"], kind="stable")
    parts: list[str] = []
    prev_type: str | None = None
    for row in group.itertuples(index=False):
        text = _clean_text(row.text)
        if not text:
            continue
        if row.sentence_type == "title":
            if not parts or parts[-1] != text:
                parts += [text, ""]
            prev_type = "title"
        elif row.sentence_type == "abstract":
            if prev_type != "abstract":
                parts.append("Abstract")
            parts += [text, ""]
            prev_type = "abstract"
        else:
            parts += [text, ""]
            prev_type = "p"
    return "\n".join(parts).strip()


# article_pairs → version_id 별 복원 본문 (version_id, text, num_sent_rows).
def build_version_text_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    sent = build_version_sentence_df(pairs_df)
    rows = []
    for version_id, group in sent.groupby("version_id", sort=False):
        text = _compose_version_text(group)
        if text:
            rows.append({"version_id": version_id, "text": text, "num_sent_rows": len(group)})
    return pd.DataFrame(rows, columns=["version_id", "text", "num_sent_rows"])


# ── version → forum 연결 + 제출시각 ────────────────────────────────────────

# mapping.references 로 version_id → forum_id fallback 표 생성 (forum 자체도 initial version 후보).
def build_mapping_version_forum_df(mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in mapping_df.itertuples(index=False):
        forum_id = _clean_text(row.id_forum)
        if forum_id:
            rows.append({"version_id": forum_id, "forum_from_mapping": forum_id})
        for version_id in _parse_listlike(row.references):
            if version_id:
                rows.append({"version_id": version_id, "forum_from_mapping": forum_id})
    out = pd.DataFrame(rows, columns=["version_id", "forum_from_mapping"])
    return out.drop_duplicates(subset=["version_id"], keep="first").reset_index(drop=True)


# version_text 에 forum_id 와 version_ts(cdate 우선, 결측 tcdate) 를 붙인다.
# forum: metadata.forum 우선 → mapping.references fallback → 최후 version_id.
def build_version_meta_df(
    version_text_df: pd.DataFrame, mapping_df: pd.DataFrame, metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata_df.copy()
    meta["version_id"] = meta["id"].map(_clean_text)
    meta["forum_from_meta"] = meta["forum"].map(_clean_text)
    ts = pd.to_numeric(meta["cdate"], errors="coerce") if "cdate" in meta else pd.Series(np.nan, index=meta.index)
    if "tcdate" in meta.columns:
        ts = ts.fillna(pd.to_numeric(meta["tcdate"], errors="coerce"))
    meta["version_ts"] = ts

    link = build_mapping_version_forum_df(mapping_df)
    out = (version_text_df
           .merge(meta[["version_id", "forum_from_meta", "version_ts"]], on="version_id", how="left")
           .merge(link, on="version_id", how="left"))

    out["forum_id"] = out["forum_from_meta"]
    for fallback in ("forum_from_mapping", "version_id"):
        blank = out["forum_id"].isna() | (out["forum_id"] == "")
        out.loc[blank, "forum_id"] = out.loc[blank, fallback]
    return out[["version_id", "text", "forum_id", "version_ts", "num_sent_rows"]]


# ── cross-venue dedup (서지 + 문자열 신호 + 사람 판정) ──────────────────────

# NFKD + ascii 변환: 합자(ﬂ→fl)·악센트(é→e) 로 인한 문자열 불일치 제거.
def _nfkd(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


# 제목+초록 시작부의 지문: boiler/‘Abstract’ 줄 제거 → NFKD → 소문자 [a-z0-9] 만 → 앞 n자.
def _prefix(text: str, n: int = 200) -> str:
    lines = [ln for ln in text.split("\n")
             if ln.strip() and not BOILER.search(ln) and ln.strip().lower() != "abstract"]
    return re.sub(r"[^a-z0-9]", "", _nfkd(" ".join(lines)).lower())[:n]


# 두 문자열의 Levenshtein 편집거리 (행 1개만 유지하는 O(min) 공간 DP).
def _levenshtein(s: str, t: str) -> int:
    m, n = len(s), len(t)
    if m == 0 or n == 0:
        return max(m, n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (s[i - 1] != t[j - 1]))
            prev = cur
    return dp[n]


# 경로압축 Union-Find (연결요소 = 같은 논문 cluster).
class UnionFind:
    def __init__(self, items) -> None:
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# 신호 A: prefix Levenshtein 유사도 ≥ EDIT_TAU 인 forum 쌍 목록 (블록=앞 18자로 후보 제한, 전수비교 회피).
def signal_a_pairs(forum_text: dict[str, str]) -> list[tuple[str, str]]:
    forums = list(forum_text)
    pref = {f: _prefix(forum_text[f]) for f in forums}
    blocks: dict[str, list[str]] = defaultdict(list)
    for f in forums:
        if len(pref[f]) >= 18:
            blocks[pref[f][:18]].append(f)
    pairs: list[tuple[str, str]] = []
    for members in blocks.values():
        for i in range(len(members)):
            a = members[i]
            for j in range(i + 1, len(members)):
                b = members[j]
                if not pref[a] or not pref[b]:
                    continue
                similarity = 1 - _levenshtein(pref[a], pref[b]) / max(len(pref[a]), len(pref[b]))
                if similarity >= EDIT_TAU:
                    pairs.append((a, b))
    return pairs


# forum 들을 cross-venue 중복 제거 → {forum_id: cluster_id}. cluster_id = 대표(가장 이른 forum) id.
# 신호 A: prefix Levenshtein ≥ EDIT_TAU.  B: paperhash 정확일치.  + MANUAL_MERGES.
def dedup_forums(
    forum_text: dict[str, str],
    forum_paperhash: dict[str, str],
    forum_date: dict[str, object],
) -> dict[str, str]:
    forums = list(forum_text)
    uf = UnionFind(forums)

    # 신호 A: prefix 편집유사도.
    for a, b in signal_a_pairs(forum_text):
        uf.union(a, b)

    # 신호 B: 같은 paperhash → 병합.
    by_hash: dict[str, list[str]] = defaultdict(list)
    for f in forums:
        ph = forum_paperhash.get(f)
        if ph:
            by_hash[ph].append(f)
    for members in by_hash.values():
        for other in members[1:]:
            uf.union(members[0], other)

    # 사람 판정 41쌍 (양쪽 모두 존재할 때만).
    for a, b in MANUAL_MERGES:
        if a in uf.parent and b in uf.parent:
            uf.union(a, b)

    # cluster 대표 = 가장 이른(날짜 최소) forum. 날짜 결측은 맨 뒤로.
    BIG = 1 << 62

    def date_key(f: str) -> tuple:
        d = forum_date.get(f)
        return (BIG if d is None else d, f)

    components: dict[str, list[str]] = defaultdict(list)
    for f in forums:
        components[uf.find(f)].append(f)
    cluster_of: dict[str, str] = {}
    for members in components.values():
        rep = min(members, key=date_key)
        for f in members:
            cluster_of[f] = rep
    return cluster_of


# version_text + mapping + metadata 에서 forum 단위 dedup 신호를 뽑는다.
# 반환: forum_text, forum_date(tcdate>cdate>pdate 최소), forum_paperhash, forum_split.
def build_forum_signals(
    version_text_df: pd.DataFrame, mapping_df: pd.DataFrame, metadata_df: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    text_by_version = dict(zip(version_text_df["version_id"].map(_clean_text), version_text_df["text"]))
    # forum 대표 텍스트 = mapping.references 순서(최신-우선)에서 '텍스트가 있는 첫 version' 본문.
    # 원본 dedup 캐시와 동일한 선택이어야 signal A(prefix) 결과가 재현됨 (경계쌍 민감).
    forum_text: dict[str, str] = {}
    for forum, refs in zip(mapping_df["id_forum"], mapping_df["references"]):
        forum = _clean_text(forum)
        if not forum or forum in forum_text:
            continue
        for version_id in _parse_listlike(refs):
            if version_id in text_by_version:
                forum_text[forum] = str(text_by_version[version_id])
                break

    meta = metadata_df.copy()
    meta["fid"] = meta["forum"].map(_clean_text)

    # forum 날짜 = 그 forum 버전들의 최소 tcdate(원본 제출시각). tcdate 결측 시 cdate/pdate.
    def _forum_date(group: pd.DataFrame) -> object:
        for col in ("tcdate", "cdate", "pdate"):
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values):
                    return int(values.min())
        return None

    forum_date = meta[meta["fid"] != ""].groupby("fid").apply(_forum_date, include_groups=False).to_dict()

    meta["ph"] = meta["content"].map(_paperhash)
    ph = meta[(meta["fid"] != "") & (meta["ph"] != "")].drop_duplicates("fid")
    forum_paperhash = dict(zip(ph["fid"], ph["ph"]))

    forum_split = dict(zip(mapping_df["id_forum"].map(_clean_text), mapping_df.get("split")))
    return forum_text, forum_date, forum_paperhash, forum_split


# ── cluster 결합 → docs_df (순수 함수, HF 불필요) ───────────────────────────

# version_meta + (forum→cluster, forum→split) → dedup cluster 단위 docs_df.
# cluster 내 전 version 을 version_ts 오름차순으로 정렬해 version_index 1..n 재부여.
def _assemble_docs_df(
    version_meta_df: pd.DataFrame, cluster_of: dict[str, str], split_of: dict[str, str],
) -> pd.DataFrame:
    df = version_meta_df.copy()
    # forum 이 clusters 에 없으면 자기 자신을 cluster 로 (singleton).
    df["cluster_id"] = df["forum_id"].map(lambda f: cluster_of.get(f, f))
    df = df.sort_values(["cluster_id", "version_ts", "version_id"],
                        kind="stable", na_position="last").reset_index(drop=True)
    df["version_index"] = df.groupby("cluster_id").cumcount() + 1

    rows = []
    for row in df.itertuples(index=False):
        family_id = build_family_id("casimir", row.cluster_id)
        version_index = int(row.version_index)
        rows.append({
            "doc_id": build_doc_id(family_id, make_revision_doc_key(version_index)),
            "dataset": "casimir",
            "family_id": family_id,
            "text": row.text,
            "version_index": version_index,
            "variant_type": "original" if version_index == 1 else "revision",
            "source_doc_id": build_doc_id(family_id, "v1") if version_index > 1 else None,
            "meta_json": {
                "forum_id": row.forum_id,
                "cluster_id": row.cluster_id,
                "raw_version_id": row.version_id,
                "split": split_of.get(row.forum_id),
                "version_ts": None if pd.isna(row.version_ts) else float(row.version_ts),
                # 복원에 쓰인 문장 행 수 (지나치게 짧은 재구성 감사용).
                "num_sent_rows": None if pd.isna(row.num_sent_rows) else int(row.num_sent_rows),
            },
        })
    return normalize_docs_df(pd.DataFrame(rows))


# CASIMIR HF 전체 → dedup cluster docs_df (family=cluster, 날짜순 version). 전 과정 HF 만으로 재현.
def build_casimir_docs_df(splits: tuple[str, ...] = CASIMIR_SPLITS) -> pd.DataFrame:
    pairs_df = load_article_pairs_df(splits)
    mapping_df = load_mapping_df(splits)
    metadata_df = load_metadata_df()

    version_text_df = build_version_text_df(pairs_df)
    version_meta_df = build_version_meta_df(version_text_df, mapping_df, metadata_df)

    forum_text, forum_date, forum_paperhash, forum_split = build_forum_signals(
        version_text_df, mapping_df, metadata_df)
    cluster_of = dedup_forums(forum_text, forum_paperhash, forum_date)
    return _assemble_docs_df(version_meta_df, cluster_of, forum_split)


# ── prepared set 분할·저장 ─────────────────────────────────────────────────

# docs_df → (v1_only, revisions=이후 전부, latest_only). later version 없는 cluster 는 제외.
def split_casimir_prepared_sets(
    docs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_version = docs_df.groupby("family_id")["version_index"].max()
    eligible = max_version[max_version >= 2].index
    work = docs_df[docs_df["family_id"].isin(eligible)].copy()

    v1 = work[work["version_index"] == 1].reset_index(drop=True)
    revisions = work[work["version_index"] > 1].reset_index(drop=True)
    latest = (revisions.sort_values(["family_id", "version_index"], kind="stable")
              .groupby("family_id", as_index=False, sort=False).tail(1).reset_index(drop=True))
    return v1, revisions, latest


# 세 dedup 세트 저장: casimir_v1_only_dedup / casimir_revisions_dedup / casimir_latest_only_dedup.
def build_and_save_casimir_prepared_sets(
    prepared_dir: str | Path, splits: tuple[str, ...] = CASIMIR_SPLITS,
) -> None:
    docs_df = build_casimir_docs_df(splits=splits)
    v1, revisions, latest = split_casimir_prepared_sets(docs_df)
    save_prepared_set(v1, prepared_dir, "casimir_v1_only_dedup")
    save_prepared_set(revisions, prepared_dir, "casimir_revisions_dedup")
    save_prepared_set(latest, prepared_dir, "casimir_latest_only_dedup")
