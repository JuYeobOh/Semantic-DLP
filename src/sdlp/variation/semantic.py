"""의미 보존 지표 — 문서 단위 Agg-BERTScore (Maddela & Alva-Manchego, NAACL 2025).

문장 단위 BERTScore 를 문서 단위로 확장한다. 원본↔변형 문장을 정렬기로 그래프 정렬 →
연결요소(다대다 정렬 세그먼트)마다 BERTScore(변형측, 원본측)를 재고 세그먼트 평균을 낸다.
**높을수록 의미 보존이 잘 됨.** (어휘 변형 지표는 quality.py, 이건 의미 보존 축)

우리 설정은 (원본 O, 변형 V) 2텍스트뿐 → 논문의 reference-less 형태에 정확히 대응:
  C(complex)=원본, S(simplified)=변형, 참조 R 없음. 그래프 간선은 원본↔변형만.

정렬기(WikiBertAligner)·스코어러(RobertaBERTScorer)는 무거운 모델이라 **주입식**으로 둔다.
→ 순수 그래프 로직(build_segments)은 유사도 행렬만 받아 결정적 단위 테스트, 모델부는 mock 으로 대체.
실제 모델을 쓴 대규모 계산은 사용자가 실행(§2 역할 분담).

정렬 모델(BERT_wiki): Jiang et al. 2020 위키↔심플위키 문장정렬 BertForSequenceClassification.
  논문 저자 배포본(구글드라이브 1I43F4OMkCvTUMtTd9Ft3P0hGiQLcFjlT)을 받아 로컬 경로로 지정.
BERTScore backbone: roberta-large (논문 각주8), bert_score 패키지, idf/baseline-rescale 없음.

512 토큰 초과 세그먼트(거대 연결요소) 처리 — oversize 인자:
  "truncate"(기본): 논문 저자 코드와 동일. bert_score 가 512 로 조용히 잘라 앞부분만 반영(뒤 손실).
  "resplit": 초과 세그먼트를 문장 단위로 budget 이하 chunk 로 재포장 → 순서대로 짝지어 각각 스코어 후
             유닛 평균. 아무 내용도 버리지 않음(단 재포장 짝짓기는 임의적 → 논문과 값이 달라질 수 있음).
  정상 케이스(패러프레이즈=거의 1:1 정렬)는 세그먼트가 작아 두 모드 결과가 같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# nltk punkt(문장 분할) 리소스 확보 — 없으면 조용히 내려받음. (nltk>=3.9 는 punkt_tab)
def _ensure_punkt() -> None:
    import nltk

    for res in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{res}")
            return
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                return
            except Exception:
                continue


# 문서를 문장 리스트로. 개행은 공백으로(논문과 동일) — 문단 경계로 문장이 붙지 않게.
def split_sentences(text: str) -> list[str]:
    from nltk import sent_tokenize

    _ensure_punkt()
    return [s for s in sent_tokenize(str(text).replace("\n", " ")) if s.strip()]


# union-find: 정렬 그래프의 연결요소를 구하기 위한 최소 구현.
class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


# 정렬 안 된 단독 문장(같은 쪽·인접 인덱스)을 하나의 세그먼트로 병합.
# 논문 merge_node_groups 재현: 무근거/누락 문장을 낱개가 아니라 인접 덩어리로 묶어 평가.
def _merge_adjacent_singletons(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    indices = sorted(indices)
    runs, cur = [], [indices[0]]
    for i in indices[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return runs


# 유사도 행렬(원본×변형)에서 임계값 초과 쌍을 간선으로 → 연결요소를 (원본측, 변형측) 세그먼트로.
# 반환: [(원본측 텍스트, 변형측 텍스트)] — 변형 문장이 하나라도 든 요소만(원본만 있는 = 누락은 후보 아님).
#   sim[i][j] = P(원본 문장 i 와 변형 문장 j 가 정렬됨). 순수 함수(모델 불필요) → 결정적 테스트.
def build_segments(
    o_sents: list[str], v_sents: list[str], sim: np.ndarray, threshold: float = 0.5
) -> list[tuple[str, str]]:
    m, n = len(o_sents), len(v_sents)
    if n == 0:
        return []
    if m == 0:                                   # 원본 없음 → 변형 전체가 무근거 세그먼트 하나
        return [("", " ".join(v_sents))]

    # 노드 0..m-1 = 원본, m..m+n-1 = 변형. 임계값 초과 쌍을 union.
    dsu = _DSU(m + n)
    for i in range(m):
        for j in range(n):
            if sim[i][j] > threshold:
                dsu.union(i, m + j)

    # 연결요소별로 원본/변형 인덱스 수집.
    comps: dict[int, tuple[list[int], list[int]]] = {}
    for i in range(m):
        comps.setdefault(dsu.find(i), ([], []))[0].append(i)
    for j in range(n):
        comps.setdefault(dsu.find(m + j), ([], []))[1].append(j)

    segments: list[tuple[str, str]] = []
    lone_o: list[int] = []                       # 정렬 안 된 단독 원본 문장 모음
    lone_v: list[int] = []                       # 정렬 안 된 단독 변형 문장 모음
    for o_idx, v_idx in comps.values():
        if len(o_idx) + len(v_idx) == 1:         # 단독 노드 → 나중에 인접 병합
            (lone_o if o_idx else lone_v).extend(o_idx or v_idx)
            continue
        segments.append((                        # 다대다 정렬된 세그먼트
            " ".join(o_sents[i] for i in sorted(o_idx)),
            " ".join(v_sents[j] for j in sorted(v_idx)),
        ))

    for run in _merge_adjacent_singletons(lone_v):   # 무근거 변형(원본측 빈 문자열)
        segments.append(("", " ".join(v_sents[j] for j in run)))
    # 단독 원본(누락)은 변형 후보가 없으므로 세그먼트로 만들지 않음(논문과 동일).
    return segments


# roberta-large BERTScore F1 (idf/baseline-rescale 없음) — 논문 각주8 설정. 스코어러 주입점.
class RobertaBERTScorer:
    # 모델 로드는 lazy(첫 호출). model_type/batch_size/device 조정 가능.
    def __init__(
        self, model_type: str = "roberta-large", batch_size: int = 64, device: str | None = None
    ) -> None:
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self._tok = None
        self._max_tokens: int | None = None

    # (변형측, 원본측) 쌍 목록 → BERTScore F1 목록(대략 0~1, 높을수록 유사). refs 는 비어있지 않아야 함.
    def score_pairs(self, cands: list[str], refs: list[str]) -> list[float]:
        if not cands:
            return []
        import bert_score

        _, _, f1 = bert_score.score(
            cands, refs, lang="en", model_type=self.model_type,
            idf=False, rescale_with_baseline=False,
            batch_size=self.batch_size, device=self.device, verbose=False,
        )
        return f1.tolist()

    # 토크나이저(모델 아님) lazy 로드 — n_tokens/max_tokens 용, resplit 모드에서만 필요.
    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_type)
        return self._tok

    # bert_score 가 truncate 하는 한계(특수토큰 포함). resplit 모드의 세그먼트 오버플로 판정 기준.
    @property
    def max_tokens(self) -> int:
        if self._max_tokens is None:
            self._max_tokens = self._tokenizer().model_max_length
        return self._max_tokens

    # 텍스트의 subword 토큰 수(특수토큰 포함, 잘림 없이). resplit 재포장 계산용.
    def n_tokens(self, text: str) -> int:
        return len(self._tokenizer().encode(text, add_special_tokens=True, truncation=False))


# 정렬 모델(BERT_wiki) 로더 — 원본×변형 문장쌍 유사도 행렬 생성. 정렬기 주입점.
class WikiBertAligner:
    # model_path = 다운로드한 BERT_wiki/ 디렉터리. do_lower_case=True(논문 저자 코드와 동일).
    def __init__(
        self, model_path: str, device: str | None = None, max_length: int = 128, batch_size: int = 64
    ) -> None:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer

        from sdlp.embedding.st import resolve_device

        self._torch = torch
        self.device = resolve_device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    # 문장쌍 목록 → P(정렬=logits 인덱스0). 논문 저자 코드의 softmax(logits)[:,0] 재현.
    def _score_pairs(self, a: list[str], b: list[str]) -> list[float]:
        torch = self._torch
        probs: list[float] = []
        for k in range(0, len(a), self.batch_size):
            enc = self.tokenizer(
                a[k: k + self.batch_size], b[k: k + self.batch_size],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())
        return probs

    # 원본×변형 유사도 행렬 (len(o) × len(v)). sim[i][j] = P(o_i 와 v_j 정렬).
    def similarity_matrix(self, o_sents: list[str], v_sents: list[str]) -> np.ndarray:
        if not o_sents or not v_sents:
            return np.zeros((len(o_sents), len(v_sents)), dtype=np.float32)
        a = [o for o in o_sents for _ in v_sents]     # text_a = 원본
        b = [v for _ in o_sents for v in v_sents]     # text_b = 변형
        flat = self._score_pairs(a, b)
        return np.asarray(flat, dtype=np.float32).reshape(len(o_sents), len(v_sents))


# 문장들을 순서 유지하며 토큰 budget 이하 chunk 로 그리디 포장(재분할용). 단문 하나가 budget 초과면 홀로 둠.
def _pack_sentences(sents: list[str], n_tokens, budget: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in sents:
        t = n_tokens(s)
        if cur and cur_n + t > budget:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# 512 초과 세그먼트를 문장 단위로 재포장 → budget 이하 (원본, 변형) 서브쌍 목록. 잉여 변형 chunk 는 무근거(원본="").
#   원본/변형 chunk 를 순서대로 짝. 잉여 원본 chunk(누락)는 후보 아님 → 버림(build_segments 와 동일 취급).
def _resplit_segment(o_seg: str, v_seg: str, n_tokens, budget: int) -> list[tuple[str, str]]:
    o_chunks = _pack_sentences(split_sentences(o_seg), n_tokens, budget) if o_seg else []
    v_chunks = _pack_sentences(split_sentences(v_seg), n_tokens, budget)
    pairs: list[tuple[str, str]] = []
    for i in range(max(len(o_chunks), len(v_chunks))):
        v_chunk = v_chunks[i] if i < len(v_chunks) else ""
        if v_chunk:                              # 변형 없는 잉여 원본 chunk 는 스킵
            pairs.append((o_chunks[i] if i < len(o_chunks) else "", v_chunk))
    return pairs or [(o_seg, v_seg)]


# 세그먼트 → 유닛(=서브쌍 목록). truncate 모드는 항상 [(o,v)] 1개. resplit 모드는 budget 초과 세그먼트만 재분할.
def _segments_to_units(
    segments: list[tuple[str, str]], oversize: str, scorer, max_tokens: int | None
) -> list[list[tuple[str, str]]]:
    if oversize != "resplit" or max_tokens is None:
        return [[(o, v)] for o, v in segments]
    units = []
    for o, v in segments:
        over = scorer.n_tokens(v) > max_tokens or (bool(o) and scorer.n_tokens(o) > max_tokens)
        units.append(_resplit_segment(o, v, scorer.n_tokens, max_tokens) if over else [(o, v)])
    return units


# 문서별 유닛 목록 → 문서별 점수. 원본측 빈 쌍은 0점. 유닛 점수=서브쌍 평균, 문서 점수=유닛 평균.
#   전 문서·전 서브쌍을 한 번에 배치 스코어(스코어러 1회 호출) → GPU 효율.
def _score_docs(docs_units: list[list[list[tuple[str, str]]]], scorer) -> list[float]:
    cands, refs, at = [], [], []
    for d, units in enumerate(docs_units):
        for u, pairs in enumerate(units):
            for p, (o_seg, v_seg) in enumerate(pairs):
                if o_seg:                        # 근거 있는 변형만 스코어러로(원본측 빈 쌍=0점)
                    cands.append(v_seg)
                    refs.append(o_seg)
                    at.append((d, u, p))
    flat = scorer.score_pairs(cands, refs) if cands else []
    pair_score = {key: 0.0 for d, units in enumerate(docs_units)
                  for u, pairs in enumerate(units) for key in [(d, u, p) for p in range(len(pairs))]}
    for key, f1 in zip(at, flat):
        pair_score[key] = f1

    doc_scores = []
    for d, units in enumerate(docs_units):
        unit_scores = [
            float(np.mean([pair_score[(d, u, p)] for p in range(len(pairs))]))
            for u, pairs in enumerate(units) if pairs
        ]
        doc_scores.append(float(np.mean(unit_scores)) if unit_scores else 0.0)
    return doc_scores


# (원본, 변형) 세그먼트 정렬 후 세그먼트별 BERTScore 평균 = 문서 Agg-BERTScore. 높을수록 의미 보존↑.
#   원본측이 빈 세그먼트(무근거 변형)는 0점. 세그먼트가 없으면(둘 다 빈 문서) 0.0.
#   oversize: "truncate"(기본, 논문 그대로 512 초과 조용히 잘림) | "resplit"(초과 세그먼트를 문장 재포장해 각각 스코어).
#   max_tokens=None 이면 resplit 시 스코어러의 max_tokens(roberta-large=512) 사용.
def agg_bertscore(
    original: str, variant: str, aligner, scorer,
    threshold: float = 0.5, oversize: str = "truncate", max_tokens: int | None = None,
) -> float:
    o_sents, v_sents = split_sentences(original), split_sentences(variant)
    if not v_sents:
        return 0.0
    sim = aligner.similarity_matrix(o_sents, v_sents)
    segments = build_segments(o_sents, v_sents, sim, threshold)
    if not segments:
        return 0.0
    budget = _resolve_budget(oversize, max_tokens, scorer)
    units = _segments_to_units(segments, oversize, scorer, budget)
    return _score_docs([units], scorer)[0]


# resplit 모드면 유효 토큰 budget 을 확정(명시값 우선, 없으면 스코어러 max_tokens). truncate 면 None.
def _resolve_budget(oversize: str, max_tokens: int | None, scorer) -> int | None:
    if oversize != "resplit":
        return None
    return max_tokens if max_tokens is not None else getattr(scorer, "max_tokens", None)


# 변형셋 각 문서를 source_doc_id 로 원본과 짝지어 Agg-BERTScore DataFrame 생성(문서 단위 행).
#   스코어러는 문서 경계를 넘어 세그먼트를 한 번에 배치 계산(정렬 모델 호출은 문서별) → GPU 효율.
#   oversize/max_tokens 는 agg_bertscore 와 동일 의미(512 초과 세그먼트 처리 방식).
def pairwise_agg_bertscore_df(
    original_df: pd.DataFrame, variant_df: pd.DataFrame, aligner, scorer,
    threshold: float = 0.5, oversize: str = "truncate", max_tokens: int | None = None,
) -> pd.DataFrame:
    ref_text = dict(zip(original_df["doc_id"], original_df["text"]))
    budget = _resolve_budget(oversize, max_tokens, scorer)

    meta: list[dict] = []                        # 문서별 행 메타
    docs_units: list[list[list[tuple[str, str]]]] = []   # 문서별 유닛
    for row in variant_df.itertuples(index=False):
        ref = ref_text.get(row.source_doc_id)
        if ref is None:                          # 원본 없는 변형은 건너뜀
            continue
        o_sents, v_sents = split_sentences(ref), split_sentences(row.text)
        sim = aligner.similarity_matrix(o_sents, v_sents) if v_sents else np.zeros((len(o_sents), 0))
        segs = build_segments(o_sents, v_sents, sim, threshold) if v_sents else []
        meta.append({
            "doc_id": row.doc_id,
            "source_doc_id": row.source_doc_id,
            "family_id": row.family_id,
            "variant_type": row.variant_type,
            "variant_level": row.variant_level,
        })
        docs_units.append(_segments_to_units(segs, oversize, scorer, budget))

    for m, score in zip(meta, _score_docs(docs_units, scorer)):
        m["agg_bertscore"] = score
    return pd.DataFrame(meta)
