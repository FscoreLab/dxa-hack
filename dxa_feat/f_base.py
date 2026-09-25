"""Базовые наборы признаков: геометрия, дуги, ось и диски позвоночника, форма бедра.

Имена колонок менять нельзя: на них обучены головы.
"""

from __future__ import annotations

from dxa_feat.anatomy import (femur_landmarks, femur_profile,
                              medial_proximal_contrast, spine_centerline,
                              spine_column)
from dxa_feat.discs_nn import discs as nn_discs
from dxa_feat.geom import bright_arcs, features
from dxa_feat.registry import feature


@feature(regions=("spine", "hip"), needs=("img", "meta", "side", "mask"))
def geometry(img, meta, side, mask):
    """Геометрия силуэта, положение в кадре, яркость."""
    return features(img, meta.mm_per_px_x, meta.mm_per_px_y, side=side, mask=mask)


@feature(regions=("spine", "hip"), needs=("img_r", "mask_r", "region", "sop_uid"))
def arcs(img_r, mask_r, region, sop_uid):
    """Яркие дуги сверху кадра — физическая мера артефактов.

    На позвоночнике маска берётся от SAM 3: Otsu вбирает артефакты внутрь кости.
    Геометрию по этой маске не считать: она узкая, без отростков, угол оси рушится.
    """
    import os
    if region == "spine" and os.environ.get("DXA_ARCS_SAM3", "1") == "1":
        m = _sam3_mask(sop_uid, img_r.shape)
        if m is None and os.environ.get("DXA_SAM3_LIVE", "1") == "1":
            # снимка нет в предрасчёте (новые данные) — иначе молча упадём на Otsu
            from dxa_seg.sam3 import column_mask
            m = column_mask(img_r)
        if m is not None:
            mask_r = m
    return bright_arcs(img_r, mask_r)


_SAM3 = None


def _sam3_mask(sop_uid, shape):
    """Маска столба от SAM 3, если посчитана заранее."""
    global _SAM3
    import numpy as np
    from pathlib import Path
    if _SAM3 is None:
        p = Path(__file__).resolve().parents[1] / "data" / "spine_sam3.npz"
        _SAM3 = np.load(p) if p.exists() else False
    if not _SAM3 or sop_uid is None or sop_uid not in _SAM3.files:
        return None
    n = int(np.prod(shape))
    return np.unpackbits(_SAM3[sop_uid])[:n].reshape(shape).astype(bool)


@feature(regions=("spine",), needs=("mask_r",))
def spine_axis(mask_r):
    """Ось колонны: наклон отдельно от кривизны."""
    return spine_centerline(mask_r)


@feature(regions=("spine",), needs=("discs",))
def spine_disc_heuristic(discs):
    """Межпозвонковые промежутки эвристикой по профилю яркости."""
    return dict(discs or {})


@feature(regions=("spine",), needs=("img_r", "mask_r", "discs"))
def spine_vertebrae(img_r, mask_r, discs):
    """Признаки по колонне позвонков, без таза и рёбер."""
    return spine_column(img_r, mask_r, (discs or {}).get("_discs"))


@feature(regions=("spine",), needs=("img_r", "mask_r", "fold"))
def spine_disc_net(img_r, mask_r, fold):
    """Промежутки обученным детектором.

    При замере кадр фолда k считает сеть, не видевшая фолда k; в поставке fold=None.
    DXA_DISC_FOLD=0 отключает это — только чтобы измерить цену утечки.
    """
    import os
    if os.environ.get("DXA_DISC_FOLD", "1") != "1":
        fold = None
    return nn_discs(img_r, mask_r, fold=fold)


@feature(regions=("hip",), needs=("mask_r",))
def femur_shape(mask_r):
    """Профиль бедра вдоль оси диафиза; все бёдра приведены к правому."""
    return femur_profile(mask_r, "right")


@feature(regions=("hip",), needs=("img_r", "mask_r"))
def femur_medial(img_r, mask_r):
    """Контраст медиальной зоны — по нему видно разворот бедра."""
    return medial_proximal_contrast(img_r, mask_r)


@feature(regions=("hip",), needs=("mask_r",))
def femur_points(mask_r):
    """Опорные точки: вертел, шейка, край диафиза."""
    return femur_landmarks(mask_r, "right")


# Отключено: на вложенной валидации только вредит.
@feature(regions=("spine",), needs=("img_r", "mask_r"), enabled=False,
         note="на вложенной валидации только вредит; охват лучше даёт fill/bbox_aspect")
def spine_segments_rejected(img_r, mask_r):
    """Признаки по отдельным телам позвонков."""
    from dxa_feat.anatomy import spine_segments
    return spine_segments(img_r, mask_r)


@feature(regions=("spine",), needs=("sop_uid", "img_r"))
def spine_axis_bodies(sop_uid, img_r):
    """Наклон оси по центрам крайних тел позвонков относительно вертикали кадра.

    Тела выделяет SAM 3. В отличие от col_tilt_deg, центр тела не зависит от
    ротации позвонков при сколиозе: отростки не тянут его вбок.
    """
    import os
    import pandas as pd
    from pathlib import Path

    global _VB
    if _VB is None:
        # DXA_VB_CSV подменяет таблицу углов целиком
        name = os.environ.get("DXA_VB_CSV", "spine_axis_bodies.csv")
        p = Path(__file__).resolve().parents[1] / "data" / name
        _VB = pd.read_csv(p).set_index("sop_uid") if p.exists() else False
    if _VB is False or sop_uid is None or sop_uid not in _VB.index:
        # снимка нет в таблице — считаем на лету, иначе правило получит -1
        if os.environ.get("DXA_SAM3_LIVE", "1") == "1" and img_r is not None:
            from dxa_seg.sam3 import body_axis
            r = body_axis(img_r)
            if r is not None:
                return {"vb_axis_deg": r[0], "vb_n": r[1]}
        return {"vb_axis_deg": -1.0, "vb_n": 0.0}
    r = _VB.loc[sop_uid]
    return {"vb_axis_deg": float(r.vb_axis_deg), "vb_n": float(r.vb_n)}


_VB = None
