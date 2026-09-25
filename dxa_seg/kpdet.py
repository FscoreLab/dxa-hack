"""Центры тел позвонков: SAM 3 + LoRA + отбор цепочкой; только инференс.

Отдельный экземпляр SAM 3: LoRA меняет DETR-часть, а базовый нужен dxa_seg.sam3.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_MODELS = Path(__file__).resolve().parents[1] / "models"
LORA = _MODELS / "kp_sam3_lora.pt"
CHAIN = _MODELS / "kp_chain.json"
SAM3 = os.environ.get("DXA_SAM3_PATH",
                      str(Path(__file__).resolve().parents[1] / ".hfcache" / "sam3"))

PITCH = 52.0          # априорный шаг тел в пикселях нашего прибора
TOPK, POOL_THR = 30, 0.05
# Минимальный промежуток соседей цепочки, в шагах: без него цепочка набирает лишние узлы.
MIN_GAP = 0.55
W0 = dict(w_s=1.0, w_a=0.5, bonus=1.0, w_p=1.0, w_c=1.5, w_d=2.0)
ATTN = ("q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "qkv", "proj")

_NET = None
_LOAD_ERROR = None


def _lora_cls():
    import torch.nn as nn

    class LoRA(nn.Module):
        # имена base/a/b — часть формата весов kp_sam3_lora.pt
        def __init__(self, base, r=8, alpha=16):
            super().__init__()
            self.base = base
            self.a = nn.Linear(base.in_features, r, bias=False)
            self.b = nn.Linear(r, base.out_features, bias=False)
            self.s = alpha / r

        def forward(self, x):
            return self.base(x) + self.b(self.a(x)) * self.s

    return LoRA


def _wrap(root) -> None:
    import torch.nn as nn
    LoRA = _lora_cls()
    for mod in root.modules():
        for name, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and name in ATTN:
                setattr(mod, name, LoRA(ch).to(ch.weight.device, ch.weight.dtype))


def _load():
    global _NET, _LOAD_ERROR
    if _NET is None:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
            if not LORA.exists():
                raise FileNotFoundError(f"нет адаптера {LORA}")
            proc = Sam3Processor.from_pretrained(SAM3, local_files_only=True)
            net = Sam3Model.from_pretrained(SAM3, local_files_only=True).eval()
            sd = torch.load(LORA, map_location="cpu", weights_only=False)
            _wrap(net.detr_decoder)
            _wrap(net.detr_encoder)
            net.load_state_dict(sd["net"], strict=False)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            net = net.to(dev)
            # Веса из файла целиком, без подмешивания W0: в kp_chain.json нет w_d, он должен быть 0.
            w = ({k: v for k, v in json.loads(CHAIN.read_text()).items()
                  if not k.startswith("_")} if CHAIN.exists() else dict(W0))
            _NET = SimpleNamespace(net=net, proc=proc, soft=sd["soft"].to(dev),
                                   dev=dev, w=w)
        except Exception as exc:
            _LOAD_ERROR = f"детектор тел недоступен ({type(exc).__name__}: {exc})"
            _NET = False
            if os.environ.get("DXA_SAM3_REQUIRED", "0") != "1":
                warnings.warn(_LOAD_ERROR, RuntimeWarning, stacklevel=2)
    if _NET is False and os.environ.get("DXA_SAM3_REQUIRED", "0") == "1":
        raise RuntimeError(_LOAD_ERROR or "детектор тел недоступен")
    return _NET or None


def _queries(d, img: np.ndarray) -> np.ndarray:
    """200 кандидатов: (N, 6) — x, y, w, h, логит, сила."""
    import torch
    from PIL import Image
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    px = d.proc(images=[Image.fromarray(np.stack([u8] * 3, -1))],
                return_tensors="pt")["pixel_values"].to(d.dev)
    te = SimpleNamespace(pooler_output=d.soft)
    am = torch.ones(1, d.soft.shape[1], dtype=torch.long, device=d.dev)
    with torch.no_grad():
        # bf16 — как при обучении адаптера; на CPU без autocast
        if d.dev == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = d.net(pixel_values=px, text_embeds=te, attention_mask=am)
        else:
            out = d.net(pixel_values=px, text_embeds=te, attention_mask=am)
    h, w = img.shape[:2]
    g = d.proc.post_process_object_detection(out, threshold=-1.0, target_sizes=[(h, w)])[0]
    b = np.asarray(g["boxes"].float().cpu())
    sc = np.asarray(g["scores"].float().cpu())
    lg = np.asarray(out.pred_logits[0].float().cpu())
    n = min(len(b), len(sc), len(lg))
    b, sc, lg = b[:n], sc[:n], lg[:n]
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], lg, sc], 1).astype(np.float32)


def _prepare(q: np.ndarray) -> np.ndarray:
    q = q[q[:, 5] >= POOL_THR]
    if len(q) > TOPK:
        q = q[np.argsort(-q[:, 5])[:TOPK]]
    return q[np.argsort(q[:, 1])] if len(q) else q


def _chain(q: np.ndarray, pitch: float, w: dict) -> np.ndarray:
    """Лучшая цепочка ДП второго порядка: индексы кандидатов сверху вниз."""
    n = len(q)
    if n < 2:
        return np.arange(n)
    xy = q[:, :2]
    V = xy[None, :, :] - xy[:, None, :]
    L = np.linalg.norm(V, axis=2)
    U = V / np.maximum(L, 1e-6)[:, :, None]
    ang = np.arccos(np.clip(np.einsum("ijd,jkd->ijk", U, U), -1, 1))
    area = q[:, 2] * q[:, 3]
    apen = np.abs(np.log(np.maximum(area, 1e-6) / ((0.9 * pitch) * (1.4 * pitch))))
    ok = (xy[None, :, 1] - xy[:, None, 1]) > MIN_GAP * pitch
    un = w["w_s"] * q[:, 4] - w["w_a"] * apen + w["bonus"]
    step = -w["w_p"] * np.abs(L - pitch) / pitch
    NEG = -1e18
    best = np.where(ok, un[:, None] + un[None, :] + step, NEG)
    # штраф за ИЗМЕНЕНИЕ шага: поясница к крестцу сужается плавно
    dch = np.abs(L[None, :, :] - L[:, :, None]) / np.maximum(L[:, :, None], 1e-6)
    cost3 = un[None, None, :] + step[None, :, :] - w["w_c"] * ang - w.get("w_d", 0.0) * dch
    cost3 = np.where(ok[:, :, None] & ok[None, :, :], cost3, NEG)
    back = np.full((n, n), -1, int)
    for k in range(n):  # кандидаты отсортированы по y — это топологический порядок
        cand = best + cost3[:, :, k]
        i_best = np.argmax(cand, axis=0)
        v = cand[i_best, np.arange(n)]
        upd = v > best[:, k]
        best[upd, k] = v[upd]
        back[upd, k] = i_best[upd]
    if best.max() <= NEG / 2:
        return np.array([int(np.argmax(q[:, 4]))])
    i, j = np.unravel_index(int(np.argmax(best)), best.shape)
    out = [j, i]
    while back[i, j] >= 0:
        i, j = back[i, j], i
        out.append(i)
    idx = np.array(sorted(set(out)), int)
    return idx[np.argsort(xy[idx, 1])]


def centers(img: np.ndarray, pitch: float = PITCH) -> np.ndarray | None:
    """Центры тел [x, y] в пикселях кадра, сверху вниз; None без детектора."""
    d = _load()
    if d is None:
        return None
    c = _prepare(_queries(d, img))
    if len(c) == 0:
        return np.zeros((0, 2))
    return c[_chain(c, pitch, d.w)][:, :2].astype(float)
