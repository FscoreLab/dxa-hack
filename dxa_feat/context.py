"""Контекст кадра: то, что наборы признаков просят через needs; тяжёлое — лениво.

`img`/`mask` — кадр как снят, `img_r`/`mask_r` — отзеркалено под правое бедро.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameContext:
    img: np.ndarray
    mask: np.ndarray | None
    meta: object
    side: str
    region: str
    embedding: np.ndarray | None = None
    # Для наборов, берущих заранее посчитанную маску по идентификатору кадра.
    sop_uid: str | None = None
    # В замере сеть берётся та, что фолда не видела; в бою None — ансамбль.
    fold: int | None = None
    _img_r: np.ndarray | None = field(default=None, repr=False)
    _mask_r: np.ndarray | None = field(default=None, repr=False)
    _discs: object = field(default=None, repr=False)
    _discs_done: bool = False
    _kp: object = field(default=None, repr=False)
    _kp_done: bool = False

    def _mirror(self, a):
        return a[:, ::-1].copy() if self.side == "left" else a

    @property
    def img_r(self) -> np.ndarray:
        if self._img_r is None:
            self._img_r = self._mirror(self.img)
        return self._img_r

    @property
    def mask_r(self) -> np.ndarray:
        """Маска кости в правой ориентации; без готовой берётся Otsu."""
        if self._mask_r is None:
            if self.mask is None:
                from dxa_seg.region import bone_mask
                self._mask_r = bone_mask(self.img_r)
            else:
                self._mask_r = self._mirror(self.mask)
        return self._mask_r

    @property
    def discs(self):
        """Межпозвонковые промежутки эвристикой; None вне позвоночника."""
        if not self._discs_done:
            self._discs_done = True
            if self.region == "spine":
                from dxa_feat.anatomy import spine_discs
                self._discs = spine_discs(self.img_r, self.mask_r)
        return self._discs

    @property
    def kp_points(self):
        """Центры тел позвонков детектором; фолдов нет — он учился на внешних данных."""
        if not self._kp_done:
            self._kp_done = True
            if self.region == "spine":
                from dxa_seg.kpdet import centers
                self._kp = centers(self.img)
        return self._kp
