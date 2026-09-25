"""Непрерывная оценка годности: замер и сервис обязаны считать её одним этим кодом.

Оценка — ранги по опорным распределениям обучающей части из `models/quality_refs.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Вес головы годности позвоночника — одно число на замер и сервис.
SPINE_QUALITY_W = 0.3

REFS_PATH = Path(__file__).resolve().parents[1] / "models" / "quality_refs.json"

# Имена — часть формата quality_refs.json; thr_* — пороги типов позвоночника (массив из одного числа).
REF_KEYS = ("spine_agg", "spine_head", "hip_head", "thr_tilt", "thr_arc", "thr_cov")


def rank(v, ref) -> np.ndarray | float:
    """Доля опорных значений строго меньше v; соглашение менять нельзя — сдвинет шкалу."""
    ref = np.asarray(ref, dtype=float)
    scalar = np.isscalar(v) or (np.ndim(v) == 0)
    x = np.atleast_1d(np.asarray(v, dtype=float))
    out = np.searchsorted(ref, x) / max(len(ref), 1)
    return float(out[0]) if scalar else out


def spine_tilt(vb: float, col: float, axis: float) -> float:
    """Наклон оси: по центрам тел, иначе по яркости, иначе по маске; -1 — не посчитан.

    Единственный источник угла для правила, калибровки порога и оценки годности.
    """
    for v in (vb, col, axis):
        if v is not None and np.isfinite(v) and v >= 0:
            return float(v)
    return -1.0


def spine_agg(tilt, arc, cov, thr: dict) -> np.ndarray:
    """Максимум по трём типам в долях порога (1 — на границе); ранги тут не годятся — не знают границы."""
    t = np.clip(np.atleast_1d(np.asarray(tilt, dtype=float)), 0.0, None)
    return np.maximum.reduce([
        t / max(float(thr["thr_tilt"][0]), 1e-9),
        np.atleast_1d(np.asarray(arc, dtype=float)) / max(float(thr["thr_arc"][0]), 1e-9),
        np.atleast_1d(np.asarray(cov, dtype=float)) / max(float(thr["thr_cov"][0]), 1e-9),
    ])


def spine_score(tilt, arc, cov, head, refs: dict, w: float = SPINE_QUALITY_W):
    """Годность позвоночника: ранг максимума по типам, смешанный с рангом головы."""
    agg = np.atleast_1d(rank(spine_agg(tilt, arc, cov, refs), refs["spine_agg"]))
    out = (1.0 - w) * agg + w * np.atleast_1d(rank(head, refs["spine_head"]))
    return float(out[0]) if np.ndim(tilt) == 0 else out


def hip_score(head, refs: dict):
    """Годность бедра — ранг своей головы, без примеси типов."""
    return rank(head, refs["hip_head"])


def load_refs(path: str | Path | None = None) -> dict | None:
    """Опорные распределения обучающей части; None, если файла нет."""
    p = Path(path or REFS_PATH)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    refs = {k: np.asarray(d[k], dtype=float) for k in REF_KEYS if k in d}
    if len(refs) != len(REF_KEYS) or any(len(v) == 0 for v in refs.values()):
        return None
    return refs


def save_refs(refs: dict, path: str | Path | None = None) -> Path:
    p = Path(path or REFS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {k: [round(float(x), 6) for x in np.sort(np.asarray(refs[k], dtype=float))]
         for k in REF_KEYS},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return p
