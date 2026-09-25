"""Эмбеддинг кадра энкодером SAM 3 — основа классификатора области.

Вектор снимается по маске кости, а не по кадру: иначе он описывает режим съёмки — шорткат.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PCA = None
# Самый грубый уровень пирамиды SAM 3: 256 каналов, крупный контекст.
FPN_LEVEL = 2


def _segmenter():
    """Энкодер — тот же экземпляр SAM 3, что меряет ось; недоступность не должна быть тихой."""
    from dxa_seg import sam3 as S3
    try:
        ok = S3._load()
    except Exception as exc:
        ok = False
        import warnings
        warnings.warn(f"энкодер эмбеддингов недоступен ({type(exc).__name__}: {exc})",
                      RuntimeWarning, stacklevel=2)
    return S3 if ok else False


def _pca():
    """PCA эмбеддинга, обученная на обучающей выборке."""
    global _PCA
    if _PCA is None:
        import pickle
        p = Path(__file__).resolve().parents[1] / "models" / "emb_pca.pkl"
        _PCA = pickle.load(open(p, "rb")) if p.exists() else False
    return _PCA


def embedding(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Нормированный вектор признаков по области кости; None, если энкодер недоступен."""
    import torch
    from PIL import Image

    S3 = _segmenter()
    if not S3:
        return None
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    px = S3._PROC(images=Image.fromarray(np.stack([u8] * 3, -1)),
                  return_tensors="pt")["pixel_values"]
    px = px.to(next(S3._MODEL.parameters()).device)
    with torch.no_grad():
        fmap = S3._MODEL.get_vision_features(pixel_values=px).fpn_hidden_states[FPN_LEVEL][0]
    _, fh, fw = fmap.shape
    small = torch.tensor(
        np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((fw, fh))) > 127,
        device=fmap.device)
    if small.sum() < 4:
        small = torch.ones_like(small)
    v = fmap[:, small].mean(1).float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-6)


def components(vec: np.ndarray | None) -> dict:
    """Вектор → несколько главных компонент под именами emb0…embN."""
    pca = _pca()
    if vec is None or not pca:
        return {}
    z = pca.transform(vec.reshape(1, -1))[0]
    return {f"emb{i}": float(v) for i, v in enumerate(z)}
