"""Определение анатомической области по изображению.

Тегов области в данных нет. Признаки — по Otsu-маске кости (SAM местами обводит
всё поле); главный — вертикальная периодичность позвонков.
"""

from __future__ import annotations

import cv2
import numpy as np

REGIONS = ("spine", "hip_single", "hip_both")


def bone_mask(img: np.ndarray, levels: int = 3) -> np.ndarray:
    """Маска кости трёхуровневым Otsu: двухуровневый режет между фоном и мягкими тканями."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    try:
        from skimage.filters import threshold_multiotsu

        th = threshold_multiotsu(u8, classes=levels)
        m = (u8 > th[-1]).astype(np.uint8) * 255
    except Exception:
        _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Верхний класс — только кортекс; разрыв между стенками диафиза горизонтальный,
    # поэтому смыкаем широким горизонтальным ядром до заливки.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    from scipy.ndimage import binary_fill_holes

    m = (binary_fill_holes(m > 0).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m > 0


def _row_periodicity(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Сила и период автокорреляции построчного профиля яркости кости."""
    prof = (img * mask).sum(axis=1)
    if prof.std() < 1e-6:
        return 0.0, 0.0
    p = prof - prof.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = max(3, int(0.08 * len(p))), max(5, int(0.45 * len(p)))
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi]
    i = int(np.argmax(seg))
    return float(seg[i]), float((lo + i) / len(p))


def features(img: np.ndarray, sam_mask: np.ndarray | None = None) -> dict:
    m = bone_mask(img)
    h, w = img.shape
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {k: 0.0 for k in (
            "area_frac", "bbox_aspect", "col_span", "profile_dip", "lr_symmetry",
            "period_strength", "period_lag", "top_heavy", "sam_area_frac", "sam_full_frame")}

    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    col = m.sum(axis=0).astype(float)
    col_s = cv2.GaussianBlur(col.reshape(-1, 1), (1, 11), 0).ravel()

    inner = col_s[int(0.25 * w):int(0.75 * w)]
    dip = float(1.0 - inner.min() / inner.max()) if len(inner) and inner.max() > 0 else 0.0
    left, right = col_s[: w // 2].sum(), col_s[w // 2:].sum()
    sym = float(min(left, right) / max(left, right)) if max(left, right) > 0 else 0.0
    strength, lag = _row_periodicity(img, m)

    # Доля массы кости в верхней трети: у бедра таз сверху тяжелее диафиза.
    top = float(m[: h // 3].sum() / max(m.sum(), 1))

    return {
        "area_frac": float(m.mean()),
        "bbox_aspect": float(bh / bw),
        "col_span": float(bw / w),
        "profile_dip": dip,
        "lr_symmetry": sym,
        "period_strength": strength,
        "period_lag": lag,
        "top_heavy": top,
        "sam_area_frac": float(sam_mask.mean()) if sam_mask is not None else 0.0,
        "sam_full_frame": float(sam_mask.mean() > 0.85) if sam_mask is not None else 0.0,
    }


def guess(f: dict) -> tuple[str, float]:
    """(область, уверенность) по эвристическим порогам."""
    # Позвоночник: выраженная периодичность плюс вытянутость по вертикали.
    spine_score = 0.6 * min(f["period_strength"] / 0.35, 1.0) + 0.4 * min(f["bbox_aspect"] / 1.6, 1.0)
    if spine_score > 0.62:
        return "spine", round(min(spine_score, 1.0), 2)
    # Два бедра: широкий охват, просвет между ними, симметрия половин.
    if f["col_span"] > 0.78 and f["profile_dip"] > 0.3 and f["lr_symmetry"] > 0.55:
        return "hip_both", round(min(f["profile_dip"] + 0.2, 1.0), 2)
    return "hip_single", round(max(0.3, 1.0 - spine_score), 2)


def side(img: np.ndarray) -> str:
    """Сторона одиночного бедра: таз смещён к своей стороне относительно диафиза."""
    m = bone_mask(img)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return "unknown"
    top = xs[ys < np.percentile(ys, 30)]
    bottom = xs[ys > np.percentile(ys, 70)]
    if len(top) == 0 or len(bottom) == 0:
        return "unknown"
    return "right" if top.mean() > bottom.mean() else "left"
