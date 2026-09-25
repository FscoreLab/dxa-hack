"""Тела позвонков L1–L4 сетью, обученной на внешнем наборе.

Наших меток сеть не видела, поэтому веса одни на все фолды и на поставку.
Приведение кадра (`prepare`) обязано совпадать с обучением.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = (384, 288)          # высота, ширина — как при обучении
N_CLS = 5                  # фон + L1..L4
_MODELS = Path(__file__).resolve().parents[1] / "models"
# Ансамбль допустим и в замере: обучающая выборка внешняя (у femur_mask — нельзя).
WEIGHTS = sorted(_MODELS.glob("vert_r34_f*.pt")) or [_MODELS / "vert_r34.pt"]

_NET = None


def _load():
    """Ленивая загрузка: без весов признаки просто не считаются."""
    global _NET
    if _NET is None:
        import torch

        from dxa_seg.unet_r34 import UNetR34
        nets = []
        for w in WEIGHTS:
            if not w.exists():
                continue
            net = UNetR34(N_CLS, pretrained=False)  # веса целиком из файла; без этого лезет в сеть за ImageNet
            net.load_state_dict(torch.load(w, map_location="cpu")["state"])
            nets.append(net.eval())
        _NET = nets or False
    return _NET


def column_x(v: np.ndarray) -> float:
    """Горизонтальное положение колонны: там кость, значит светло."""
    col = cv2.GaussianBlur(v, (0, 0), 3).mean(0)
    k = max(5, len(col) // 10)
    return float(np.argmax(np.convolve(col, np.ones(k) / k, mode="same")))


def prepare(v: np.ndarray) -> np.ndarray:
    """Кадр -> вход сети: масштаб по высоте, кроп по ширине вокруг колонны."""
    H, W = SIZE
    h, w = v.shape
    k = H / h
    z = cv2.resize(v, (max(1, int(round(w * k))), H), interpolation=cv2.INTER_LINEAR)
    cx = int(round(column_x(v) * k))
    x0 = max(0, min(cx - W // 2, z.shape[1] - W)) if z.shape[1] >= W else 0
    out = np.zeros((H, W), np.float32)
    piece = z[:, x0:x0 + W]
    out[:, : piece.shape[1]] = piece
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def drop_islands(body: np.ndarray, min_frac: float = 0.08, gap: int = 14) -> np.ndarray:
    """Убрать куски (крыло таза, ребро), оторванные от основной колонны."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), 8)
    if n <= 1:
        return body
    areas = st[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    near = cv2.dilate((lab == main).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))) > 0
    keep = np.zeros(body.shape, bool)
    for i in range(1, n):
        sel = lab == i
        if i == main or (areas[i - 1] >= min_frac * areas[main - 1] and (sel & near).any()):
            keep |= sel
    return keep


def keep_core(lab: np.ndarray, min_frac: float = 0.25) -> np.ndarray:
    """В каждом уровне оставить основной кусок, мелкие спутники отбросить."""
    out = np.zeros_like(lab)
    for c in range(1, N_CLS):
        m = (lab == c).astype(np.uint8)
        if not m.any():
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
        areas = st[1:, cv2.CC_STAT_AREA]
        for i in range(1, n):
            if areas[i - 1] >= min_frac * areas.max():
                out[cc == i] = c
    return out


def by_rows(prob: np.ndarray) -> np.ndarray:
    """Уровень назначается полосе строк, порядок L1→L4 сверху вниз обязателен.

    Попиксельный argmax дробит уровни, а соседние тела слипаются в одну компоненту.
    """
    body = drop_islands(prob[1:].sum(0) > prob[0])
    score = np.stack([np.where(body, prob[c], 0).sum(1) for c in range(1, N_CLS)], 1)
    H, K = score.shape
    f = np.zeros((H + 1, K))
    bk = np.zeros((H + 1, K), np.int8)
    for y in range(H):
        best, arg = -np.inf, 0
        for c in range(K):
            # Накапливается индекс максимума, а не максимум индексов — иначе уровень не сменится.
            if f[y, c] > best:
                best, arg = f[y, c], c
            f[y + 1, c] = best + score[y, c]
            bk[y + 1, c] = arg
    out = np.zeros(prob.shape[1:], np.uint8)
    c = int(np.argmax(f[H]))
    lvl = np.zeros(H, np.uint8)
    for y in range(H, 0, -1):
        lvl[y - 1] = c + 1
        c = int(bk[y, c])
    for y in range(H):
        out[y][body[y]] = lvl[y]
    return keep_core(out)


def levels(img: np.ndarray) -> np.ndarray | None:
    """Кадр -> карта уровней (0 фон, 1..4 = L1..L4) в приведённых координатах."""
    net = _load()
    if net is False:
        return None
    import torch

    z = prepare(np.asarray(img, dtype=np.float32))
    x = torch.from_numpy(z[None, None])
    with torch.no_grad():
        prob = np.mean([torch.softmax(n(x), 1)[0].numpy() for n in net], 0)
    return by_rows(prob)
