"""Измерения поясницы по цепочке центров тел: крыло таза и тонкие яркие структуры.

Параметры подобраны вручную на обучающей выборке и в пикселях прибора 0,6 × 1,05 мм:
на другом аппарате их нужно пересчитать.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

HALF_W = 35.0      # полуширина тела, px по x
GAP = 1.3          # коридор колонны, в полуширинах
WING_REL = 0.35    # кость крыла — ярче этой доли яркости тел
HAT_K, HAT_T = 7, 0.25


def interp_x(p: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """x оси на строках ys: линейно между центрами, за концами — по двум крайним."""
    if len(p) == 1:
        return np.full(len(ys), p[0, 0])
    x = np.interp(ys, p[:, 1], p[:, 0])
    lo, hi = ys < p[0, 1], ys > p[-1, 1]
    k0 = (p[1, 0] - p[0, 0]) / max(p[1, 1] - p[0, 1], 1e-6)
    k1 = (p[-1, 0] - p[-2, 0]) / max(p[-1, 1] - p[-2, 1], 1e-6)
    x[lo] = p[0, 0] + k0 * (ys[lo] - p[0, 1])
    x[hi] = p[-1, 0] + k1 * (ys[hi] - p[-1, 1])
    return x


def body_level(img: np.ndarray, p: np.ndarray, r: int = 3) -> float:
    """Типичная яркость тела: медиана окон вокруг центров."""
    v = []
    for x, y in p:
        x, y = int(round(x)), int(round(y))
        v.append(img[max(0, y - r):y + r + 1, max(0, x - 2 * r):x + 2 * r + 1].ravel())
    return float(np.median(np.concatenate(v))) if v else float(np.median(img))


def wing_top(img: np.ndarray, p: np.ndarray, pitch: float) -> tuple[float, float | None]:
    """(высота крыла в шагах тел, верхняя строка или None); NaN — засвет выше 0,6 кадра."""
    h, w = img.shape
    ref = body_level(img, p)
    sm = ndi.gaussian_filter(img, 1.0)
    d = np.arange(w)[None, :] - interp_x(p, np.arange(h, dtype=float))[:, None]
    wmax = 0.0
    for lat in (d <= -GAP * HALF_W, d >= GAP * HALF_W):
        lab, _ = ndi.label((sm >= WING_REL * ref) & lat)
        touch = np.unique(lab[h - 3:][lab[h - 3:] > 0])
        m = np.isin(lab, touch) if len(touch) else np.zeros_like(lat)
        rows = np.nonzero(m.sum(1) >= 0.3 * HALF_W)[0]
        wmax = max(wmax, h - (float(rows.min()) if len(rows) else float(h)))
    if 0 < wmax < 0.6 * h:
        return wmax / pitch, h - wmax
    return (0.0 if wmax == 0 else float("nan")), None


def artifact_mask(img: np.ndarray, p: np.ndarray, y_stop: float | None) -> np.ndarray:
    """Тонкие яркие структуры вне колонны; колонну исключает цепочка, не маска Otsu."""
    import cv2
    h, w = img.shape
    ref = max(body_level(img, p), 1e-3)
    n = np.clip(img / ref, 0, 3).astype(np.float32)
    cx = interp_x(p, np.arange(h, dtype=float))
    zone = np.abs(np.arange(w)[None, :] - cx[:, None]) >= GAP * HALF_W
    if y_stop is not None:
        zone[int(y_stop):] = False
    # замаскированные лаборантом чёрные поля — не ткань
    zone &= ndi.binary_erosion(img > 0.02, np.ones((5, 5)))
    hat = cv2.morphologyEx(n, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAT_K, HAT_K)))
    return (hat >= HAT_T) & zone
