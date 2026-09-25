"""Детектор межпозвонковых промежутков: профиль, архитектура, разбор выхода.

В пакете, а не в scripts/: его грузит dxa_feat.discs_nn в инференсе.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dxa_feat.anatomy import _largest_component

L = 256           # длина нормированного профиля
SIGMA = 3.0       # ширина гауссианы в целевой карте, в отсчётах профиля
RANDOM_STATE = 20260916


def profile(img, mask):
    """Профиль яркости вдоль колонны плюс её полуширина — два канала."""
    col = _largest_component(mask)
    h, w = col.shape
    vals, half = [], []
    ys = [y for y in range(h) if col[y].any()]
    if len(ys) < 20:
        return None, None, None
    for y in range(h):
        xs = np.nonzero(col[y])[0]
        if len(xs):
            c = int(0.5 * (xs.min() + xs.max()))
            r = max(3, int(0.40 * 0.5 * (xs.max() - xs.min())))
            vals.append(img[y, max(0, c - r): min(w, c + r + 1)].mean())
            half.append(0.5 * (xs.max() - xs.min()))
        else:
            vals.append(np.nan)
            half.append(np.nan)
    y0, y1 = ys[0], ys[-1]
    v = np.array(vals[y0:y1 + 1], dtype=np.float32)
    hw = np.array(half[y0:y1 + 1], dtype=np.float32)
    v = np.nan_to_num(v, nan=float(np.nanmedian(v)))
    hw = np.nan_to_num(hw, nan=float(np.nanmedian(hw)))
    v = (v - v.mean()) / (v.std() + 1e-6)
    hw = (hw - hw.mean()) / (hw.std() + 1e-6)
    return v, hw, (y0, y1)


def to_len(a, n=L):
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


class DiscNet(nn.Module):
    """Свёртки с растущим полем зрения: диск виден только в контексте соседей."""

    def __init__(self, ch=32):
        super().__init__()
        layers, c_in = [], 2
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d),
                       nn.BatchNorm1d(ch), nn.ReLU()]
            c_in = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)


def peaks(pred, thr=0.3, min_gap=8):
    out = []
    for i in range(1, len(pred) - 1):
        if pred[i] >= pred[i - 1] and pred[i] > pred[i + 1] and pred[i] > thr:
            if not out or i - out[-1] >= min_gap:
                out.append(i)
            elif pred[i] > pred[out[-1]]:
                out[-1] = i
    return out


