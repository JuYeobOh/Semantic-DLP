"""PAR3 병렬 번역 데이터셋 (par3.pkl) → 문서 단위 정규 docs_df.

한 작품 = 단락(para)들을 줄바꿈으로 이어붙인 문서 1개.
- par3_gt       : GoogleMT 전문 (기밀 DB, original)
- par3_human    : 사람 번역가별 전문 (positive, human_translation) — 작품당 2~5명
- par3_human_t1 : 사람 번역가 1번(translator_1)만
family_id = work_id (같은 작품의 gt + 모든 사람번역이 한 family).

ponytail: par3 는 bespoke id(work_id 를 family_id 로 직접, "par3:" 접두 없음).
"""
from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from sdlp.io import save_prepared_set
from sdlp.schemas import normalize_docs_df

# work_id 끝의 _xx 언어코드.
_LANG_RE = re.compile(r"_([a-z]{2})$")
# work_id 중간의 _<권번호>_ (분권 표시).
_VOL_RE = re.compile(r"_(\d+)_")
_LANG_NAME = {
    "cs": "Czech", "de": "German", "es": "Spanish", "fa": "Persian",
    "fr": "French", "hu": "Hungarian", "it": "Italian", "ja": "Japanese",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "ta": "Tamil", "zh": "Chinese",
}


# work_id 에서 (제목슬러그, 언어코드) 분리.
def _split_lang(work_id: str) -> tuple[str, str]:
    m = _LANG_RE.search(work_id)
    return (work_id[: m.start()], m.group(1)) if m else (work_id, "unknown")


# 단락 리스트를 줄바꿈으로 결합 (빈 단락 제외).
def _join(paras: list[str]) -> str:
    return "\n".join(s for p in paras if (s := (p or "").strip()))


# 분권(같은 base 의 여러 권)을 권 순서대로 이어붙여 하나의 work 로 병합.
# 단독 1권·비분권 work 는 그대로 둔다. 반환: {work_id -> work dict(gt_paras/source_paras/translator_data)}.
def _merge_volumes(data: dict) -> dict:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    singles: list[str] = []
    for wid in sorted(data):
        m = _VOL_RE.search(wid)
        if m:
            groups[wid[: m.start()]].append((int(m.group(1)), wid))
        else:
            singles.append(wid)

    result: dict[str, dict] = {wid: data[wid] for wid in singles}
    for base, items in groups.items():
        if len(items) == 1:                 # 단독 1권(ponniyin) → 원본 유지
            result[items[0][1]] = data[items[0][1]]
            continue
        items.sort()                        # 권 번호 순
        _, lang = _split_lang(items[0][1])
        translators = sorted(data[items[0][1]]["translator_data"].keys())
        gt, source = [], []
        tparas: dict[str, list] = defaultdict(list)
        for _, wid in items:
            w = data[wid]
            gt += w["gt_paras"]
            source += w["source_paras"]
            for tk in translators:
                tparas[tk] += w["translator_data"][tk]["translator_paras"]
        result[f"{base}_{lang}"] = {
            "gt_paras": gt,
            "source_paras": source,
            "translator_data": {tk: {"translator_paras": tparas[tk]} for tk in translators},
        }
    return result


# par3.pkl → (gt docs_df, human docs_df). 분권은 병합 후 처리.
def build_par3_frames(pkl_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(pkl_path, "rb") as f:
        works = _merge_volumes(pickle.load(f))

    gt_rows: list[dict] = []
    human_rows: list[dict] = []
    for work_id, work in sorted(works.items()):
        _, lang = _split_lang(work_id)
        translators = sorted(work["translator_data"].keys())
        meta = {
            "lang": lang,
            "lang_name": _LANG_NAME.get(lang, lang),
            "n_paragraphs": len(work["gt_paras"]),
            "n_translators": len(translators),
        }
        gt_rows.append({
            "doc_id": f"{work_id}__gt",
            "dataset": "par3",
            "family_id": work_id,
            "text": _join(work["gt_paras"]),
            "version_index": 0,
            "variant_type": "original",
            "source_doc_id": None,
            "meta_json": {**meta, "role": "machine_translation"},
        })
        for vi, tk in enumerate(translators, start=1):
            text = _join(work["translator_data"][tk]["translator_paras"])
            if not text:
                continue
            human_rows.append({
                "doc_id": f"{work_id}__{tk}",
                "dataset": "par3",
                "family_id": work_id,
                "text": text,
                "version_index": vi,
                "variant_type": "human_translation",
                "variant_level": tk,
                "source_doc_id": f"{work_id}__gt",
                "meta_json": {**meta, "translator": tk, "role": "human_translation"},
            })
    return normalize_docs_df(pd.DataFrame(gt_rows)), normalize_docs_df(pd.DataFrame(human_rows))


# par3_gt / par3_human / par3_human_t1(translator_1만) 세트 저장.
def build_and_save_par3_prepared_sets(pkl_path: str | Path, prepared_dir: str | Path) -> None:
    gt_df, human_df = build_par3_frames(pkl_path)
    human_t1 = human_df[human_df["variant_level"] == "translator_1"].reset_index(drop=True)
    save_prepared_set(gt_df, prepared_dir, "par3_gt")
    save_prepared_set(human_df, prepared_dir, "par3_human")
    save_prepared_set(human_t1, prepared_dir, "par3_human_t1")
