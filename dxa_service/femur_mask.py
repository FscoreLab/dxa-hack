"""Маска бедра в инференсе: лёгкая U-Net, обученная на псевдометках SAM, работает на CPU."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = 128
_NETS: list | None = None


def _load():
    """Ансамбль фолдовых сетей; элемент k обучен без фолда k — на этом держится femur_mask(fold=k)."""
    global _NETS
    if _NETS is None:
        import torch

        from dxa_seg.unet import UNet

        root = Path(__file__).resolve().parents[1] / "models"
        paths = sorted(root.glob("femur_unet_*.pt")) or [root / "femur_unet.pt"]
        nets = []
        for p in paths:
            if not p.exists():
                continue
            n = UNet()
            n.load_state_dict(torch.load(p, map_location="cpu"))
            n.eval()
            nets.append(n)
        _NETS = nets or False
    return _NETS


def femur_mask(img: np.ndarray, side: str = "",
               fold: int | None = None) -> np.ndarray | None:
    """Маска бедренной кости; None без весов. Левый кадр зеркалится: сеть обучена на правом бедре.

    fold=k — для замера: одна сеть, не видевшая фолда k. Ансамбль в замере завысил бы оценку.
    """
    import torch

    net = _load()
    if not net:
        return None
    if fold is not None:
        if fold >= len(net):
            raise IndexError(
                f"фолд {fold}, а сетей {len(net)}: в models/ должны лежать "
                f"femur_unet_0..4.pt, иначе честный замер невозможен")
        net = [net[fold]]
    a = img[:, ::-1].copy() if side == "left" else img
    x = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    t = torch.tensor(x)[None, None].float()
    with torch.no_grad():
        # Усреднение по сетям ансамбля и по отражению (TTA).
        acc = None
        for n in net:
            p1 = torch.sigmoid(n(t))
            p2 = torch.flip(torch.sigmoid(n(torch.flip(t, [3]))), [2])
            q = (0.5 * (p1 + p2)).numpy()[0]
            acc = q if acc is None else acc + q
        p = acc / len(net)
    m = cv2.resize((p > 0.5).astype(np.uint8), (a.shape[1], a.shape[0]),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    # Постобработка как у псевдометок: провалы контраста в шейке — не полости.
    u8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Заливка от рамки, а не от (0,0): если маска касается угла, костью объявится весь кадр.
    pad = cv2.copyMakeBorder(u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 1)
    out = (u8 | (1 - ff[1:-1, 1:-1])).astype(bool)
    return out[:, ::-1].copy() if side == "left" else out
