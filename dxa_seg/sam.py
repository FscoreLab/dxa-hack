"""Предразметка кости через SAM 2.1 — только подготовка данных, в сервис не идёт."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

MODEL_ID = "facebook/sam2.1-hiera-large"


@dataclass
class BoneMask:
    mask: np.ndarray            # bool, размер исходного кадра
    n_components: int
    area_frac: float
    raggedness: float           # периметр / периметр выпуклой оболочки; 1.0 — гладкий


def bright_prompt(img: np.ndarray, pct: float = 85.0) -> tuple[list[float], list[float]]:
    """Рамка и центроид крупнейшей компоненты Otsu (центр яркого пятна попадает в остистый отросток)."""
    from dxa_seg.region import bone_mask

    m = bone_mask(img)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = lab == big
        point = [float(cent[big][0]), float(cent[big][1])]
    else:
        ys, xs = np.nonzero(img >= np.percentile(img, pct))
        point = [float(xs.mean()), float(ys.mean())]
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        h, w = img.shape
        return [0.0, 0.0, float(w - 1), float(h - 1)], [w / 2.0, h / 2.0]
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    # центроид может лечь на тёмный просвет — подтягиваем к ближайшему пикселю маски
    if not m[int(round(point[1])), int(round(point[0]))]:
        d = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
        k = int(np.argmin(d))
        point = [float(xs[k]), float(ys[k])]
    return box, point


def _raggedness(mask: np.ndarray) -> tuple[int, float]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.arcLength(cv2.convexHull(c), True)
    return len(cnts), cv2.arcLength(c, True) / max(hull, 1.0)


class BoneSegmenter:
    """Ленивая обёртка над SAM 2.1: модель грузится один раз на процесс."""

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        from transformers import Sam2Model, Sam2Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    def segment(self, img: np.ndarray, point: list[float] | None = None,
                box: list[float] | None = None) -> BoneMask:
        """Рамка + точка, крупнейший кандидат: по одной точке SAM берёт остистый отросток."""
        from PIL import Image

        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if point is None or box is None:
            b, p = bright_prompt(img)
            box = box or b
            point = point or p
        inputs = self.processor(
            images=Image.fromarray(np.stack([u8] * 3, -1)),
            input_boxes=[[box]],
            input_points=[[[point]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        cand = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0][0].numpy() > 0
        mask = cand[int(np.argmax([c.sum() for c in cand]))] if cand.ndim == 3 else cand
        n, ragged = _raggedness(mask)
        return BoneMask(mask=mask, n_components=n, area_frac=float(mask.mean()), raggedness=ragged)


    def segment_points(self, img: np.ndarray, points: list[list[float]],
                       labels: list[int]) -> BoneMask:
        """Сегментация по точкам (1 — объект, 0 — фон), минимальный валидный кандидат.

        Без рамки (на бедре она захватывает таз) и не по score (самый уверенный — вся анатомия).
        """
        from PIL import Image

        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        inputs = self.processor(
            images=Image.fromarray(np.stack([u8] * 3, -1)),
            input_points=[[points]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        cand = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0][0].numpy() > 0
        if cand.ndim == 2:
            cand = cand[None]

        h, w = img.shape
        pos = [p for p, l in zip(points, labels) if l == 1]
        neg = [p for p, l in zip(points, labels) if l == 0]

        def ok(m):
            at = lambda p: m[min(h - 1, max(0, int(round(p[1])))),
                             min(w - 1, max(0, int(round(p[0]))))]
            return all(at(p) for p in pos) and not any(at(p) for p in neg)

        valid = [m for m in cand if ok(m) and m.mean() > 0.01]
        mask = min(valid, key=lambda m: m.sum()) if valid else cand[0]
        n, ragged = _raggedness(mask)
        return BoneMask(mask=mask, n_components=n, area_frac=float(mask.mean()), raggedness=ragged)


def overlay(img: np.ndarray, mask: np.ndarray, label: str = "") -> np.ndarray:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    vis = np.stack([u8] * 3, -1).copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 70, 70), 1)
    if label:
        cv2.putText(vis, label, (3, 11), cv2.FONT_HERSHEY_PLAIN, 0.7, (90, 220, 255), 1, cv2.LINE_AA)
    return vis
