"""Охват и посторонние предметы по цепочке центров тел (детектор dxa_seg.kpdet)."""

import numpy as np

from dxa_feat.kp_measure import artifact_mask, wing_top
from dxa_feat.registry import feature


@feature(regions=("spine",), needs=("img", "kp_points"))
def kp_spine(img, kp_points):
    """Видимое крыло таза и доля кадра под тонкими яркими структурами вне колонны."""
    p = kp_points
    if p is None or len(p) < 2:
        # нечем мерить — не добавляем нарушений на пустом месте
        return {"kp_wing_vis": 1.0, "kp_wing_absent": 0.0, "kp_art_frac": 0.0}
    img = np.asarray(img, dtype=float)
    pitch = float(np.median(np.diff(p[:, 1])))
    vis, top = wing_top(img, p, pitch)
    # NaN — «крыло» выше 0,6 кадра, засвет мягких тканей: считаем таз видимым
    vis = 1.0 if not np.isfinite(vis) else vis
    art = artifact_mask(img, p, top)
    return {"kp_wing_vis": float(vis), "kp_wing_absent": float(vis <= 0),
            "kp_art_frac": float(art.sum()) / img.size}
