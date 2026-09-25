"""Сеть ориентиров бедра: U-Net с головами маски и пяти карт точек.

⚠ Живёт в пакете, а не в scripts/: её грузит сервис (dxa_feat/f_hipmarks.py).
"""

import torch.nn as nn

from dxa_seg.unet import UNet

POINTS = ["малый вертел: пик выступа внутрь", "большой вертел: верхушка",
          "шейка: верхний край перетяжки", "шейка: нижний край перетяжки",
          "седалищная кость: нижний край"]


class MultiNet(nn.Module):
    """U-Net с двумя головами: маска и пять карт ориентиров."""

    def __init__(self, c=16, n_pts=len(POINTS)):
        super().__init__()
        self.body = UNet(c)
        self.body.out = nn.Identity()          # снимаем однослойный выход
        self.head_mask = nn.Conv2d(c, 1, 1)
        self.head_pts = nn.Conv2d(c, n_pts, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head_mask(z).squeeze(1), self.head_pts(z)
