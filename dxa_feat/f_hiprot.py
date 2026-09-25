"""Ротация бедра по яркости зоны малого вертела, а не по силуэту.

При ротации вертел проецируется внутрь кости: по контуру маски его не видно, по плотности — видно.
"""
from __future__ import annotations

import numpy as np

from dxa_feat.registry import feature


def _axis(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    y_lo, y_hi = int(ys.min()), int(ys.max())
    span = max(1, y_hi - y_lo)
    low = mask.copy()
    low[: int(y_lo + 0.80 * span)] = False
    lys, lxs = np.nonzero(low)
    if len(lxs) < 10:
        return None
    return y_lo, span, float(lxs.mean()), float(lys.mean())


@feature(regions=("hip",), needs=("img_r", "mask_r"))
def lesser_trochanter_density(img_r, mask_r):
    """Малый вертел как плотность; медиальная сторона развёрнутого кадра — правая."""
    m = np.asarray(mask_r, dtype=bool)
    a = _axis(m)
    if a is None or m.sum() < 200:
        return {"lt_dens_empty": 1.0}
    y_lo, span, sx, sy = a
    img = np.asarray(img_r, dtype=np.float32)
    h, w = m.shape

    # полоса малого вертела: ниже шейки, верхняя половина кости
    y0, y1 = int(y_lo + 0.25 * span), int(y_lo + 0.55 * span)
    band = np.zeros_like(m)
    band[y0:y1] = True
    med = band & m
    if med.sum() < 40:
        return {"lt_dens_empty": 1.0}

    # опора: плотность диафиза, чтобы мера не зависела от пациента
    low = m.copy()
    low[: int(y_lo + 0.80 * span)] = False
    ref = float(np.median(img[low])) if low.sum() > 20 else float(np.median(img[m]))
    ref = max(ref, 1e-3)

    rows = []
    for y in range(y0, y1):
        xs = np.nonzero(m[y])[0]
        if len(xs) < 6:
            continue
        # медиальная (правая) половина строки: вертел проецируется сюда
        x0, x1 = int(xs.max() - 0.5 * (xs.max() - xs.min())), int(xs.max())
        if x1 <= x0 + 1:
            continue
        seg = img[y, x0:x1 + 1]
        rows.append((float(seg.max()), float(seg.mean()),
                     float((len(seg) - 1 - np.argmax(seg)) / max(len(seg) - 1, 1))))
    if len(rows) < 5:
        return {"lt_dens_empty": 1.0}
    mx = np.array([r[0] for r in rows])
    mn = np.array([r[1] for r in rows])
    pos = np.array([r[2] for r in rows])
    return {
        "lt_dens_empty": 0.0,
        # растёт и когда вертел торчит наружу, и когда наложился — оба полюса ротации
        "lt_dens_peak": float(mx.max() / ref),
        "lt_dens_mean": float(mn.mean() / ref),
        # от медиального края: 0 — вертел снаружи, 1 — наложился на кость
        "lt_dens_pos": float(np.median(pos)),
        "lt_dens_pos_std": float(np.std(pos)),
        # резкость границы вертела: при наложении контур размывается
        "lt_dens_grad": float(np.median(np.abs(np.diff(mx)))) / ref,
    }
