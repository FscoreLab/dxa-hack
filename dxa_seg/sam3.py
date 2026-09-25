"""Сегментация SAM 3 по текстовому запросу на лету, для нового снимка.

Без весов при DXA_SAM3_REQUIRED=1 — ошибка, иначе запасной путь с предупреждением.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

MODEL = os.environ.get(
    "DXA_SAM3_PATH",
    str(Path(__file__).resolve().parents[1] / ".hfcache" / "sam3"))
PROMPT_COLUMN = "vertebral bodies of the lumbar spine"
PROMPT_BODY = "rectangular vertebral body"

_MODEL = None
_PROC = None
_LOAD_ERROR = None


def _load():
    global _MODEL, _PROC, _LOAD_ERROR
    if _MODEL is None:
        try:
            if not Path(MODEL).is_dir():
                raise FileNotFoundError(f"каталог весов SAM 3 не найден: {MODEL}")
            import torch
            from transformers import Sam3Model, Sam3Processor
            _PROC = Sam3Processor.from_pretrained(MODEL, local_files_only=True)
            m = Sam3Model.from_pretrained(MODEL, local_files_only=True).eval()
            _MODEL = m.cuda() if torch.cuda.is_available() else m
        except Exception as exc:
            _LOAD_ERROR = f"SAM 3 недоступен ({type(exc).__name__}: {exc})"
            _MODEL = False
            if os.environ.get("DXA_SAM3_REQUIRED", "0") != "1":
                warnings.warn(_LOAD_ERROR + "; ось будет измерена запасным методом",
                              RuntimeWarning, stacklevel=2)
    if _MODEL is False and os.environ.get("DXA_SAM3_REQUIRED", "0") == "1":
        raise RuntimeError(_LOAD_ERROR or "SAM 3 недоступен")
    return _MODEL is not False


def _masks(img: np.ndarray, prompt: str):
    if not _load():
        return None
    import torch
    from PIL import Image

    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    inp = _PROC(images=Image.fromarray(np.stack([u8] * 3, -1)), text=prompt,
                return_tensors="pt")
    if torch.cuda.is_available():
        inp = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inp.items()}
    with torch.no_grad():
        out = _MODEL(**inp)
    g = _PROC.post_process_instance_segmentation(
        out, threshold=0.3, mask_threshold=0.5,
        target_sizes=[(img.shape[0], img.shape[1])])[0]
    mk = g.get("masks")
    if mk is None or len(mk) == 0:
        return None
    mk = np.asarray(mk.cpu() if hasattr(mk, "cpu") else mk).astype(bool)
    return mk[None] if mk.ndim == 2 else mk


def column_mask(img: np.ndarray):
    """Маска столба позвонков одним куском — под признак артефактов."""
    mk = _masks(img, PROMPT_COLUMN)
    if mk is None:
        return None
    keep = [mk[i] for i in range(mk.shape[0]) if mk[i].sum() >= 60]
    return np.logical_or.reduce(keep) if keep else None


def body_centers(img: np.ndarray) -> list[tuple[float, float]] | None:
    """Центры тел позвонков, (y, x), сверху вниз."""
    mk = _masks(img, PROMPT_BODY)
    if mk is None:
        return None
    cents = []
    for i in range(mk.shape[0]):
        m = mk[i]
        if m.sum() < 60:
            continue
        ys, xs = np.nonzero(m)
        cents.append((float(ys.mean()), float(xs.mean())))
    return sorted(cents) if len(cents) >= 2 else None


def axis_deg(cents, signed: bool = False) -> float:
    """Угол оси по крайним телам от вертикали; знак «+» — нижнее тело правее (нужен синтетике)."""
    (ty, tx), (by, bx) = cents[0], cents[-1]
    a = float(np.degrees(np.arctan((bx - tx) / max(by - ty, 1e-6))))
    return a if signed else abs(a)


def body_axis(img: np.ndarray):
    """Угол оси по центрам крайних тел и число найденных тел."""
    cents = body_centers(img)
    if cents is None:
        return None
    return axis_deg(cents), float(len(cents))
