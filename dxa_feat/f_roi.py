"""Боковые запасы вокруг области интереса бедра в миллиметрах (ТЗ: 2 см от края).

Масштаб только горизонтальный, паспортные 0,6 мм: вертикальный спорный.
"""

import numpy as np

from dxa_feat.registry import feature

MM_X = 0.6
SIDE_MARGIN_MM = 20.0     # «2 см от края» — ТЗ, с. 6


# Выключен: запас от силуэта до края кадра — конвенция кадрирования, не дефект укладки.
@feature(regions=("hip",), needs=("mask", "meta"), enabled=False,
         note="боковой запас в мм: ухудшает годность и охват ROI")
def hip_side_margins(mask, meta):
    """Боковые запасы кости до края кадра в миллиметрах и флаги критерия."""
    m = np.asarray(mask, dtype=bool) if mask is not None else None
    if m is None or not m.any():
        return {"roi_side_empty": 1.0}
    ys, xs = np.nonzero(m)
    w = m.shape[1]
    mm = float(getattr(meta, "mm_per_px_x", None) or MM_X)
    left = float(xs.min()) * mm
    right = float(w - 1 - xs.max()) * mm
    return {
        "roi_side_empty": 0.0,
        "roi_margin_left_mm": left,
        "roi_margin_right_mm": right,
        # критерий про каждый край — минимум, а не среднее
        "roi_margin_min_mm": min(left, right),
        "roi_margin_ok": float(min(left, right) >= SIDE_MARGIN_MM),
        "roi_margin_deficit_mm": max(0.0, SIDE_MARGIN_MM - min(left, right)),
    }
