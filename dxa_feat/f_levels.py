"""Нумерация уровней позвонков и положение колонны в кадре.

Критерий охвата сформулирован именами позвонков (Th12 сверху, гребни снизу).
"""

import numpy as np

from dxa_feat.anatomy import vertebra_levels
from dxa_feat.registry import feature


@feature(regions=("spine",), needs=("img_r", "mask_r", "discs"))
def spine_levels(img_r, mask_r, discs):
    """Уровни позвонков от линии Якоби; точность наследуется от детектора дисков."""
    d = (discs or {}).get("_discs") if isinstance(discs, dict) else discs
    if d is None or len(np.atleast_1d(d)) < 2:
        return {"no_levels": 1.0}
    return vertebra_levels(img_r, mask_r, list(np.atleast_1d(d)))


@feature(regions=("spine", "hip"), needs=("mask", "region"))
def frame_offset(mask, region):
    """Смещение кости от центра кадра; по исходному кадру — зеркало меняет сторону."""
    m = np.asarray(mask, dtype=bool) if mask is not None else None
    if m is None or not m.any():
        return {"cx_rel": -1.0, "cx_abs": -1.0}
    ys, xs = np.nonzero(m)
    rel = float(xs.mean()) / max(m.shape[1] - 1, 1) - 0.5
    return {"cx_rel": rel, "cx_abs": abs(rel)}
