"""Признаки укладки бедра по ориентирам мультитаск-сети — по требованиям методики.

Масштаб по Y 0,73 мм, а не паспортные 1,05: при паспортном анатомия выходит из нормы.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dxa_feat.registry import feature

MM_X, MM_Y = 0.6, 0.73
POINTS = ["малый вертел: пик выступа внутрь", "большой вертел: верхушка",
          "шейка: верхний край перетяжки", "шейка: нижний край перетяжки",
          "седалищная кость: нижний край"]
MIN_CONF = 0.20

_NETS = {}


def _nets(fold=None):
    """Сети ориентиров; fold=k — одна сеть, не видевшая фолда (без него ансамбль и утечка в замере)."""
    key = "all" if fold is None else int(fold)
    if key not in _NETS:
        _NETS[key] = []
        try:
            import torch
            from dxa_seg.multitask import MultiNet
            ks = range(5) if fold is None else [int(fold)]
            for k in ks:
                p = Path(f"models/multitask_{k}.pt")
                if p.exists():
                    n = MultiNet()
                    n.load_state_dict(torch.load(p, map_location="cpu"))
                    _NETS[key].append(n.eval())
        except Exception as exc:
            import warnings
            warnings.warn(f"сети ориентиров бедра не загрузились ({type(exc).__name__}: {exc})",
                          RuntimeWarning, stacklevel=2)
            _NETS[key] = []
    return _NETS[key]


def _predict(img_r, fold=None):
    """Ориентиры в координатах кадра; None, если сети нет."""
    nets = _nets(fold)
    if not nets:
        return None
    import cv2
    import torch
    from dxa_seg.unet import SIZE

    x = cv2.resize(np.asarray(img_r, np.float32), (SIZE, SIZE),
                   interpolation=cv2.INTER_AREA)
    acc = None
    with torch.no_grad():
        for n in nets:
            _, pp = n(torch.tensor(x[None, None]))
            p = torch.sigmoid(pp)[0].numpy()
            acc = p if acc is None else acc + p
    acc /= len(nets)
    h, w = np.asarray(img_r).shape
    out = {}
    for c, name in enumerate(POINTS):
        conf = float(acc[c].max())
        if conf < MIN_CONF:
            out[name] = None
            continue
        py, px = np.unravel_index(int(np.argmax(acc[c])), acc[c].shape)
        out[name] = (px * w / SIZE, py * h / SIZE, conf)
    return out


def medial_line(mask_r):
    """Медиальный контур диафиза x = k*y + c: (k, c, y_from, y_to) или None; общий для признака и картинки."""
    m = np.asarray(mask_r, bool)
    if m.sum() <= 200:
        return None
    ys, _ = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max())
    # в развёрнутом кадре медиальный край — правый край маски, не левый
    rows = [(y, np.nonzero(m[y])[0]) for y in range(int(y0 + 0.6 * (y1 - y0)), y1)]
    rows = [(y, r[-1]) for y, r in rows if len(r)]
    if len(rows) <= 10:
        return None
    yy = np.array([r[0] for r in rows], float)
    xx = np.array([r[1] for r in rows], float)
    k, c = np.polyfit(yy, xx, 1)
    return float(k), float(c), int(y0 + 0.45 * (y1 - y0)), y1


@feature(regions=("hip",), needs=("img_r", "mask_r", "fold"))
def hip_landmark_rules(img_r, mask_r, fold):
    """Требования методики к укладке бедра, каждое отдельным числом."""
    if os.environ.get("DXA_HIPMARKS", "1") != "1":
        return {}
    P = _predict(img_r, fold)
    if P is None:
        return {"lmk_empty": 1.0}
    img = np.asarray(img_r)
    h, w = img.shape
    f = {"lmk_empty": 0.0}

    gt = P.get("большой вертел: верхушка")
    isc = P.get("седалищная кость: нижний край")
    lt = P.get("малый вертел: пик выступа внутрь")
    nu = P.get("шейка: верхний край перетяжки")
    nd = P.get("шейка: нижний край перетяжки")

    # «не менее 3 см ткани выше большого вертела»
    if gt:
        f["lmk_above_gt_mm"] = float(gt[1] * MM_Y)
        f["lmk_above_gt_ok"] = float(gt[1] * MM_Y >= 30.0)
    # «не менее 3 см ниже седалищной кости»
    if isc:
        f["lmk_below_isch_mm"] = float((h - 1 - isc[1]) * MM_Y)
        f["lmk_below_isch_ok"] = float((h - 1 - isc[1]) * MM_Y >= 30.0)
    # «малый вертел немного выступает за внутренний край»
    f["lmk_lt_absent"] = float(lt is None)
    line = medial_line(mask_r) if lt is not None else None
    if line is not None:
        k, c = line[:2]
        # положительное — вертел выступает медиально за контур диафиза
        f["lmk_lt_out_mm"] = float((lt[0] - (k * lt[1] + c)) * MM_X)
    # Шейка в боковой проекции — только прокси: угол отрезка между краями перетяжки
    # и расстояние от вертела до её середины меняются при ротации, но не критерий.
    if nu and nd:
        dx, dy = (nu[0] - nd[0]) * MM_X, (nu[1] - nd[1]) * MM_Y
        f["lmk_neck_w_mm"] = float(np.hypot(dx, dy))
        f["lmk_neck_ang"] = float(abs(np.degrees(np.arctan2(dx, dy + 1e-6))))
        if gt:
            mid = ((nu[0] + nd[0]) / 2, (nu[1] + nd[1]) / 2)
            f["lmk_neck_len_mm"] = float(np.hypot((mid[0] - gt[0]) * MM_X,
                                                  (mid[1] - gt[1]) * MM_Y))
            if f.get("lmk_neck_w_mm", 0) > 1:
                # шейка мнимо укорачивается при ротации в обе стороны
                f["lmk_neck_ratio"] = f["lmk_neck_len_mm"] / f["lmk_neck_w_mm"]
    # Границ ROI на снимке нет: заход зоны шейки на вертел — прокси по расстоянию.
    if nu and gt:
        f["lmk_neck_to_gt_mm"] = float(np.hypot((nu[0] - gt[0]) * MM_X,
                                                 (nu[1] - gt[1]) * MM_Y))
    if gt and lt is not None:
        f["lmk_gt_lt_mm"] = float(np.hypot((gt[0] - lt[0]) * MM_X,
                                            (gt[1] - lt[1]) * MM_Y))
    return f
