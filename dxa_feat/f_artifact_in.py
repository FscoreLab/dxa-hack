"""Артефакты внутри кости: металл, вертебропластика, обызвествления.

Прочие признаки артефактов по построению обнуляются под маской кости.
"""

import cv2
import numpy as np

from dxa_feat.registry import feature


# Выключен: в разметке артефакты — предметы снаружи кости, металла внутри нет.
@feature(regions=("spine", "hip"), needs=("img_r", "mask_r"), enabled=False,
         note="в разметке нет металла внутри кости — признаку не на чем учиться")
def artifact_inside(img_r, mask_r):
    """Яркие включения внутри костной маски и их доля от площади кости."""
    img = np.asarray(img_r, dtype=float)
    m = np.asarray(mask_r, dtype=bool) if mask_r is not None else None
    if m is None or not m.any():
        return {"in_empty": 1.0}
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # Порог по кости, а не по кадру: иначе отберётся половина кортикального слоя.
    inside_hat = hat[m]
    thr = max(12, int(np.percentile(inside_hat, 99.0)))
    hot = (hat >= thr) & m
    area = float(m.sum())
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
    big, comp = 0.0, 0
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a < 4:
            continue
        comp += 1
        big = max(big, a / area)
    # Насыщение именно в кости: глобальный saturated_frac эндопротезы не выделяет.
    sat = float((img[m] >= 0.995).mean())
    return {
        "in_empty": 0.0,
        "in_hot_frac": float(hot.sum()) / area,     # доля ярких включений в кости
        "in_hot_max_frac": big,                     # самое крупное включение
        "in_hot_n": float(comp),
        "in_sat_frac": sat,                         # насыщение внутри кости
        "in_contrast": float(np.percentile(img[m], 99) - np.median(img[m])),
    }
