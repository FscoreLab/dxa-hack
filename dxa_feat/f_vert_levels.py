"""Охват и укладка по телам L1–L4 в единицах высоты тела на этом же кадре.

Маски — сеть `dxa_seg.vert_levels`, обученная на внешнем наборе; фолд не нужен.
"""

import cv2
import numpy as np

from pathlib import Path

from dxa_feat.registry import feature
from dxa_seg.vert_levels import N_CLS, SIZE, column_x, levels

MISS = -1.0


def _empty() -> dict:
    keys = ("gap_top_bodies", "gap_bot_bodies", "tilt_ends", "tilt_resid",
            "aspect_med", "aspect_slope", "width_slope", "lr_asym", "body_h_px")
    return {f"vl_{k}": MISS for k in keys}


def _external(sop_uid, img):
    """Готовая маска уровней из DXA_VL_MASKS — те же признаки на масках другой модели."""
    import os

    d = os.environ.get("DXA_VL_MASKS")
    if not d or sop_uid is None:
        return None
    f = Path(d) / f"{sop_uid}.png"
    if not f.exists():
        return None
    m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    H, W = SIZE
    h, w = np.asarray(img).shape
    k = H / h
    z = cv2.resize(m, (max(1, int(round(w * k))), H), interpolation=cv2.INTER_NEAREST)
    c = int(round(column_x(np.asarray(img, dtype=np.float32)) * k))
    x0 = max(0, min(c - W // 2, z.shape[1] - W)) if z.shape[1] >= W else 0
    out = np.zeros((H, W), m.dtype)
    piece = z[:, x0:x0 + W]
    out[:, : piece.shape[1]] = piece
    return out


@feature(regions=("spine",), needs=("img", "sop_uid"))
def vert_levels(img, sop_uid):
    """Положение и форма тел L1–L4, нормированные на высоту позвонка."""
    lab = _external(sop_uid, img)
    if lab is None:
        lab = levels(img)
    if lab is None:
        return _empty()
    H, W = lab.shape

    box, hgt, wid, cen = {}, {}, {}, {}
    for c in range(1, N_CLS):
        ys, xs = np.nonzero(lab == c)
        if len(ys) < 50:
            continue
        box[c] = (int(ys.min()), int(ys.max()))
        hgt[c] = ys.max() - ys.min() + 1
        # ширина по медиане строк, а не по габариту: отростки дают выбросы
        wid[c] = float(np.median([np.count_nonzero(lab[y] == c)
                                  for y in range(ys.min(), ys.max() + 1)]))
        cen[c] = (ys.mean(), xs.mean())
    if len(box) < 2:
        return _empty()

    ks = sorted(box)
    h = float(np.mean([hgt[c] for c in ks]))          # единица измерения
    top, bot = ks[0], ks[-1]

    # Касание края — по крайней строке кадра, а не по габариту маски.
    touch_top = np.count_nonzero(lab[0] == top) / max(wid[top], 1.0)
    touch_bot = np.count_nonzero(lab[H - 1] == bot) / max(wid[bot], 1.0)
    whole = sum(1 for c in ks
                if box[c][0] > 0 and box[c][1] < H - 1
                and not (lab[:, 0] == c).any() and not (lab[:, W - 1] == c).any())

    ys = np.array([cen[c][0] for c in ks])
    xs = np.array([cen[c][1] for c in ks])
    if len(ks) > 2:
        k, b = np.polyfit(ys, xs, 1)
        resid = float(np.abs(xs - (k * ys + b)).max() / h)   # изогнутость, не наклон
    else:
        resid = MISS
    # наклон по КРАЙНИМ точкам: на длинной колонне это работает иначе, чем регрессия
    tilt = float(np.degrees(np.arctan((xs[-1] - xs[0]) / max(ys[-1] - ys[0], 1e-6))))

    asp = np.array([hgt[c] / max(wid[c], 1.0) for c in ks])
    lvl = np.arange(len(ks), dtype=float)
    slope_a = float(np.polyfit(lvl, asp, 1)[0]) if len(ks) > 2 else MISS
    w_arr = np.array([wid[c] for c in ks]) / max(h, 1.0)
    slope_w = float(np.polyfit(lvl, w_arr, 1)[0]) if len(ks) > 2 else MISS

    # асимметрия относительно собственной оси колонны, а не центра кадра
    axis = float(np.mean(xs))
    left = float((lab[:, : int(axis)] > 0).sum())
    right = float((lab[:, int(axis) :] > 0).sum())
    asym = abs(left - right) / max(left + right, 1.0)

    # touch_top/touch_bot/whole не отдаются: на наших данных они константы.
    return {
        # сколько ПОЗВОНКОВ помещается выше L1 и ниже L4 — вопрос критерия
        "vl_gap_top_bodies": float(box[top][0] / h),
        "vl_gap_bot_bodies": float((H - 1 - box[bot][1]) / h),
        "vl_tilt_ends": tilt,
        "vl_tilt_resid": resid,
        "vl_aspect_med": float(np.median(asp)),
        "vl_aspect_slope": slope_a,
        "vl_width_slope": slope_w,
        "vl_lr_asym": float(asym),
        "vl_body_h_px": float(h / H),      # масштаб позвонка в кадре
    }
