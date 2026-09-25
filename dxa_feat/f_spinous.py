"""Смещение гребня остистых отростков от центра тел — признак ротации позвонков.

Это не угол оси: ось меряется по телам. Гребень ищется как argmax сглаженного
профиля строки (центроид размазывает отросток по телу); величины нормированы на
ширину колонны.
"""

import cv2
import numpy as np

from dxa_feat.registry import feature

MISS = -1.0
KEYS = ("offset_med", "offset_max", "offset_slope", "offset_sd",
        "ridge_conf", "width_med")


def _empty() -> dict:
    return {"sp_absent": 1.0, **{f"sp_{k}": MISS for k in KEYS}}


# img_r/mask_r, а не img/mask: `mask` в контексте может быть None; на
# позвоночнике зеркалирования нет, данные те же.
@feature(regions=("spine",), needs=("img_r", "mask_r"), enabled=False,
         note="гребень остистых отростков: ось в полном замере не улучшает")
def spinous_offset(img_r, mask_r):
    """Смещение гребня отростков от центра тел по высоте колонны."""
    m = np.asarray(mask_r, dtype=bool)
    v = np.asarray(img_r, dtype=np.float32)
    if m.sum() < 500:
        return _empty()

    # колонна — крупнейшая компонента: таз и рёбра сюда не нужны
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return _empty()
    col = lab == (int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1)

    ys = np.nonzero(col.any(1))[0]
    if len(ys) < 40:
        return _empty()

    # По одной строке пик слабый и argmax прыгает; сильное сглаживание вдоль оси
    # поднимает непрерывный по вертикали гребень и гасит зернистость.
    sm = cv2.GaussianBlur(v, (0, 0), sigmaX=1.5, sigmaY=15.0)

    lo, hi = int(ys.min() + 0.15 * len(ys)), int(ys.max() - 0.15 * len(ys))
    off, conf, wid = [], [], []
    for y in range(lo, hi + 1):
        xs = np.nonzero(col[y])[0]
        if len(xs) < 12:
            continue
        x0, x1 = xs.min(), xs.max()
        w = x1 - x0 + 1
        prof = sm[y, x0:x1 + 1].astype(np.float32)
        # только центральная половина: по краям суставные отростки и край тела ярче
        a, b = int(0.25 * w), int(0.75 * w)
        if b - a < 4:
            continue
        seg = prof[a:b]
        peak = float(a + np.argmax(seg))
        centre = (w - 1) / 2.0
        off.append((peak - centre) / w)
        rng = float(seg.max() - np.median(prof))
        conf.append(rng / (float(np.std(prof)) + 1e-6))
        wid.append(float(w))
    if len(off) < 20:
        return _empty()

    off = np.asarray(off)
    yy = np.linspace(0.0, 1.0, len(off))
    slope = float(np.polyfit(yy, off, 1)[0])   # ротация нарастает по высоте?
    return {
        "sp_absent": 0.0,
        "sp_offset_med": float(np.median(np.abs(off))),
        "sp_offset_max": float(np.percentile(np.abs(off), 90)),
        "sp_offset_slope": slope,
        "sp_offset_sd": float(np.std(off)),
        "sp_ridge_conf": float(np.median(conf)),
        "sp_width_med": float(np.median(wid) / max(v.shape[1], 1)),
    }
