"""Охват позвоночника по телам L1–L4, найденным сетью; величины в долях кадра.

Сеть обучена только на внешнем наборе, поэтому фолд не нужен.
"""

import numpy as np

from dxa_feat.registry import feature
from dxa_seg.vert_levels import N_CLS, SIZE, levels

MISS = -1.0                    # уровня нет в кадре

# Знаковым величинам MISS двусмыслен (наклон -1° законен): пропуск — по nnl_absent.
SIGNED = ("nnl_tilt_deg",)


# Выключен: сеть не видит Th12 и гребни, о которых критерий охвата; годность не растёт.
@feature(regions=("spine",), needs=("img",), enabled=False,
         note="уровни L1-L4 сетью: охват ранжируется лучше, годность нет")
def levels_nn(img):
    """Положение и пропорции тел L1–L4 в кадре."""
    lab = levels(img)
    if lab is None:
        return {"nnl_absent": 1.0, "nnl_found": MISS, "nnl_top_margin": MISS,
                "nnl_bot_margin": MISS, "nnl_l1_top": MISS, "nnl_l4_bot": MISS,
                "nnl_h_med": MISS, "nnl_h_cv": MISS, "nnl_l4_h_rel": MISS,
                "nnl_tilt_deg": MISS, "nnl_x_span": MISS, "nnl_area_frac": MISS}

    H, W = SIZE
    box, cen, hgt = {}, {}, {}
    for c in range(1, N_CLS):
        ys, xs = np.nonzero(lab == c)
        if len(ys) < 50:
            continue
        box[c] = (ys.min() / H, ys.max() / H)
        cen[c] = (ys.mean() / H, xs.mean() / W)
        hgt[c] = (ys.max() - ys.min() + 1) / H

    if not box:
        return {"nnl_absent": 1.0, "nnl_found": 0.0, "nnl_top_margin": MISS,
                "nnl_bot_margin": MISS, "nnl_l1_top": MISS, "nnl_l4_bot": MISS,
                "nnl_h_med": MISS, "nnl_h_cv": MISS, "nnl_l4_h_rel": MISS,
                "nnl_tilt_deg": MISS, "nnl_x_span": MISS, "nnl_area_frac": MISS}

    top, bot = min(box), max(box)
    h = np.array(list(hgt.values()))
    # x от y, а не наоборот: колонна почти вертикальна, обратный наклон взрывается
    ys = np.array([cen[c][0] for c in sorted(cen)])
    xs = np.array([cen[c][1] * W / H for c in sorted(cen)])
    tilt = float(np.degrees(np.arctan(np.polyfit(ys, xs, 1)[0]))) if len(ys) > 1 else MISS

    return {
        "nnl_absent": 0.0,
        "nnl_found": float(len(box)),
        "nnl_l1_top": float(box[top][0]),
        "nnl_l4_bot": float(box[bot][1]),
        "nnl_top_margin": float(box[top][0]),
        "nnl_bot_margin": float(1.0 - box[bot][1]),
        "nnl_h_med": float(np.median(h)),
        "nnl_h_cv": float(h.std() / max(h.mean(), 1e-6)),
        # L4 у сети иногда прихватывает L5: тело выходит вдвое выше прочих
        "nnl_l4_h_rel": float(hgt[bot] / max(np.median(h), 1e-6)),
        "nnl_tilt_deg": tilt,
        "nnl_x_span": float(max(cen[c][1] for c in cen) - min(cen[c][1] for c in cen)),
        "nnl_area_frac": float((lab > 0).sum() / lab.size),
    }
