"""Межпозвонковые диски обученной сетью вместо эвристики.

Без весов возвращается пустой словарь, и вызывающий код берёт эвристику.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_NETS: dict = {}
L = 256


def _net(fold: int | None = None):
    """Сеть дисков; fold=k — не видевшая фолда, без неё замер не подменяется боевой."""
    if fold in _NETS:
        return _NETS[fold]
    import os
    import torch

    root = Path(__file__).resolve().parents[1] / "models"
    # DXA_DISC_EXT=1 — сеть только на внешнем наборе: фолды не нужны, утечки нет.
    ext = root / "disc_net_ext.pt"
    if os.environ.get("DXA_DISC_EXT") == "1" and ext.exists():
        p = ext
    else:
        p = root / ("disc_net.pt" if fold is None else f"disc_net_{fold}.pt")
    if not p.exists():
        if fold is not None and p.name.startswith("disc_net_") and p.name != "disc_net_ext.pt":
            raise FileNotFoundError(
                f"нет {p.name}: честный замер требует пяти фолдовых сетей, "
                f"соберите их scripts/pipeline/train_disc_net.py --folds")
        _NETS[fold] = False
    else:
        from dxa_feat.disc_net import DiscNet
        n = DiscNet()
        n.load_state_dict(torch.load(p, map_location="cpu"))
        n.eval()
        _NETS[fold] = n
    return _NETS[fold]


def discs(img: np.ndarray, mask: np.ndarray, fold: int | None = None) -> dict:
    """Позиции дисков и производные величины; пустой словарь без весов."""
    import torch

    from dxa_feat.anatomy import _largest_component
    from dxa_feat.disc_net import peaks, profile, to_len

    net = _net(fold)
    if not net:
        return {}
    v, hw, rng = profile(img, _largest_component(mask))
    if v is None:
        return {}
    x = np.stack([to_len(v), to_len(hw)])[None]
    with torch.no_grad():
        pr = net(torch.tensor(x)).numpy()[0]
    y0, y1 = rng
    span = max(1, y1 - y0)
    d = np.array(sorted(y0 + p * span / L for p in peaks(pr)))
    if len(d) < 2:
        return {"nn_disc_n": float(len(d))}
    gaps = np.diff(d)
    step = float(np.median(gaps))
    return {
        "nn_disc_n": float(len(d)),
        "nn_step_norm": step / span,
        "nn_step_cv": float(np.std(gaps) / (step + 1e-6)),
        "nn_top_gap": float((d.min() - y0) / step),
        "nn_bot_gap": float((y1 - d.max()) / step),
        "nn_conf": float(np.mean([pr[int(round((p - y0) / span * L))]
                                  for p in d if 0 <= (p - y0) / span * L < L])),
        "_discs": d.tolist(),
    }
