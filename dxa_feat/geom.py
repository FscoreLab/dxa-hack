"""Геометрические и яркостные признаки кадра по маске кости.

Расстояния в миллиметрах, где известен масштаб: критерии ТЗ заданы в сантиметрах,
а пиксель у Lunar анизотропный и различается между режимами.
"""

from __future__ import annotations

import cv2
import numpy as np

from dxa_seg.region import bone_mask


def axis_angle_deg(mask: np.ndarray) -> float:
    """Наклон главной оси кости к вертикали, градусы 0..90.

    Знак собственного вектора произволен (arctan2 может дать ~179°), поэтому
    угол складывается как min(a, 180 − a).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    w, v = np.linalg.eigh(np.cov(pts))
    major = v[:, int(np.argmax(w))]
    a = abs(float(np.degrees(np.arctan2(major[0], major[1]))))
    a = min(a, 180.0 - a)
    return min(a, 180.0 - a)


def curvature(mask: np.ndarray) -> float:
    """Отклонение центров строк от прямой в долях ширины: сколиоз против наклона укладки."""
    ys = np.arange(mask.shape[0])
    cx = np.full(mask.shape[0], np.nan)
    for y in range(mask.shape[0]):
        xs = np.nonzero(mask[y])[0]
        if len(xs):
            cx[y] = xs.mean()
    ok = ~np.isnan(cx)
    if ok.sum() < 20:
        return 0.0
    p = np.polyfit(ys[ok], cx[ok], 1)
    resid = cx[ok] - np.polyval(p, ys[ok])
    return float(np.abs(resid).mean() / mask.shape[1])


def thin_bright_score(img: np.ndarray, mask: np.ndarray) -> dict:
    """Тонкие яркие структуры вне кости (косточки белья, клипсы, провода) по white top-hat."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k)
    outside = top.copy()
    outside[cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0
    # Otsu здесь вырождается: почти весь кадр нули, порог садится в шум
    bw = (outside >= 25).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_elong, n_thin, total = 0.0, 0, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 8:
            continue
        per = cv2.arcLength(c, True)
        elong = per ** 2 / (4 * np.pi * a + 1e-9)  # 1.0 — круг, больше — вытянутое
        total += a
        if elong > 6:
            n_thin += 1
            best_elong = max(best_elong, elong)
    h = img.shape[0]
    upper = outside[: int(0.45 * h)]
    return {"thin_n": float(n_thin), "thin_elong": float(best_elong),
            "thin_area_frac": float(total / img.size),
            "outside_bright": float(outside.mean() / 255.0),
            # косточки белья лежат на мягких тканях в верхней части спайн-кадра
            "outside_bright_upper": float(upper.mean() / 255.0),
            "thin_upper_frac": float((upper >= 25).mean())}


def anatomy_visibility(mask: np.ndarray) -> dict:
    """Видимость гребней снизу и рёбер сверху — прямые критерии охвата спайна."""
    h, w = mask.shape
    side = np.zeros_like(mask)
    side[:, : int(0.28 * w)] = True
    side[:, int(0.72 * w):] = True
    bottom = mask[int(0.85 * h):] & side[int(0.85 * h):]
    top = mask[: int(0.15 * h)] & side[: int(0.15 * h)]
    return {"iliac_bottom_frac": float(bottom.mean()), "ribs_top_frac": float(top.mean())}


def vertebra_count(img: np.ndarray, mask: np.ndarray, mm_y: float | None) -> dict:
    """Число видимых позвонков по периодичности профиля яркости.

    Масштаб берётся из паспорта аппарата; высота позвонка в мм — признак формы,
    а не источник калибровки.
    """
    prof = (img * mask).sum(axis=1)
    if prof.std() < 1e-6:
        return {"n_vertebrae": 0.0, "period_px": 0.0, "period_strength": 0.0,
                "vertebra_mm": 0.0}
    p = prof - prof.mean()
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = max(3, int(0.08 * len(p))), max(5, int(0.45 * len(p)))
    if hi <= lo:
        return {"n_vertebrae": 0.0, "period_px": 0.0, "period_strength": 0.0,
                "vertebra_mm": 0.0}
    seg = ac[lo:hi]
    k = int(np.argmax(seg))
    period = float(lo + k)
    ys = np.nonzero(mask.any(axis=1))[0]
    span = float(ys.max() - ys.min() + 1) if len(ys) else 0.0
    return {
        "n_vertebrae": span / period if period > 0 else 0.0,
        "period_px": period,
        "period_strength": float(seg[k]),
        "vertebra_mm": period * mm_y if (period > 0 and mm_y) else 0.0,
    }


def _masked_fov(img: np.ndarray) -> dict:
    """Закрашенные лаборантом области и реальное поле зрения.

    Закрашенное ровно нулевое, фон съёмки шумит — по этому и отличаем.
    """
    z = img <= 0.002
    h, w = img.shape
    colz, rowz = z.mean(0), z.mean(1)
    open_c = np.nonzero(colz <= 0.98)[0]
    open_r = np.nonzero(rowz <= 0.98)[0]
    if not len(open_c) or not len(open_r):
        return {"masked_frac": float(z.mean()), "fov_w_frac": 1.0, "fov_h_frac": 1.0,
                "masked_cols": 0.0, "masked_rows": 0.0}
    return {
        "masked_frac": float(z.mean()),                       # сколько кадра закрашено
        "fov_w_frac": float((open_c.max() - open_c.min() + 1) / w),   # ширина живого поля
        "fov_h_frac": float((open_r.max() - open_r.min() + 1) / h),
        "masked_cols": float((colz > 0.98).sum() / w),
        "masked_rows": float((rowz > 0.98).sum() / h),
    }


def features(img: np.ndarray, mm_x: float | None, mm_y: float | None,
             side: str = "", mask: np.ndarray | None = None) -> dict:
    """Признаки силуэта кости. mask — готовая маска; без неё Otsu, который на бедре сливает кость с тазом."""
    # все бёдра приводятся к правому, иначе сторона учится как отдельный класс
    if side == "left":
        img = img[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
    m = bone_mask(img) if mask is None else mask
    h, w = img.shape
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"empty": 1.0}

    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    inside = img[m]
    fov = _masked_fov(img)
    bg = img[~m]
    hi = float(img.max())
    # Доля пикселей у потолка яркости — металл и эндопротезы упираются в 255.
    saturated = float((img >= hi - 1e-6).mean())

    f = {
        "empty": 0.0,
        "area_frac": float(m.mean()),
        "bbox_aspect": (y1 - y0 + 1) / (x1 - x0 + 1),
        "fill": float(m.sum() / ((y1 - y0 + 1) * (x1 - x0 + 1))),
        "col_span": (x1 - x0 + 1) / w,
        "row_span": (y1 - y0 + 1) / h,
        # запас до краёв кадра — прямой аналог критериев охвата
        "margin_top_frac": y0 / h,
        "margin_bottom_frac": (h - 1 - y1) / h,
        "margin_left_frac": x0 / w,
        "margin_right_frac": (w - 1 - x1) / w,
        "touch_top": float(y0 <= 1),
        "touch_bottom": float(y1 >= h - 2),
        "touch_left": float(x0 <= 1),
        "touch_right": float(x1 >= w - 2),
        "axis_deg": axis_angle_deg(m),
        "curvature": curvature(m),
        "int_mean": float(inside.mean()),
        "int_std": float(inside.std()),
        "int_p95": float(np.percentile(inside, 95)),
        "contrast": float(inside.mean() - (bg.mean() if bg.size else 0.0)),
        "saturated_frac": saturated,
        "n_components": float(len(cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])),
        "bg_mean": float(bg.mean()) if bg.size else 0.0,
    }
    f.update(thin_bright_score(img, m))
    f.update(anatomy_visibility(m))
    f.update(vertebra_count(img, m, mm_y))
    # Признаки маскировки по умолчанию выключены: на замере ухудшали. Включить: DXA_FOV=1.
    import os
    if os.environ.get("DXA_FOV"):
        f.update(fov)
    # поля ещё и от живого поля зрения: лаборант мог закрасить часть кадра
    if not os.environ.get("DXA_FOV"):
        return f
    fw = max(fov["fov_w_frac"] * w, 1.0)
    fh = max(fov["fov_h_frac"] * h, 1.0)
    f["margin_top_fov"] = f["margin_top_frac"] * h / fh
    f["margin_bottom_fov"] = f["margin_bottom_frac"] * h / fh
    f["margin_left_fov"] = f["margin_left_frac"] * w / fw
    f["margin_right_fov"] = f["margin_right_frac"] * w / fw
    return f


def bright_arcs(img: np.ndarray, mask: np.ndarray) -> dict:
    """Длинные тонкие яркие дуги вне кости — провода, трубки, украшения.

    По площади они ничтожны, глобальная яркость их не ловит. Тонкость — white
    top-hat, протяжённость — длина главной оси компоненты.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    thr = max(18, int(np.percentile(hat, 99.0)))
    outside = (hat >= thr) & ~mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8), 8)
    h, w = img.shape
    best_len, total_len, n_long = 0.0, 0.0, 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs - xs.mean(), ys - ys.mean()])
        ev = np.linalg.eigvalsh(np.cov(pts)) if len(xs) > 2 else np.array([0.0, 0.0])
        length = 4.0 * float(np.sqrt(max(ev[-1], 0.0)))      # ~длина главной оси
        rel = length / max(w, 1)
        best_len = max(best_len, rel)
        total_len += rel
        if rel > 0.25:
            n_long += 1
    up = outside[: int(0.35 * h)]
    return {
        "arc_max_len": best_len,          # самая длинная дуга в долях ширины кадра
        "arc_total_len": total_len,
        "arc_n_long": float(n_long),      # сколько дуг длиннее четверти кадра
        "arc_upper_frac": float(up.mean()),
    }
