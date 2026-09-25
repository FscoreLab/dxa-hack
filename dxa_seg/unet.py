"""Архитектура U-Net для маски кости.

Живёт в пакете, а не в scripts/: её грузит сервис. Обучение — scripts/pipeline/unet.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SIZE = 128
RANDOM_STATE = 20260916


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.d1, self.d2, self.d3 = block(1, c), block(c, c * 2), block(c * 2, c * 4)
        self.bott = block(c * 4, c * 8)
        self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2); self.c3 = block(c * 8, c * 4)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2); self.c2 = block(c * 4, c * 2)
        self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2); self.c1 = block(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        a = self.d1(x); b = self.d2(self.pool(a)); c = self.d3(self.pool(b))
        z = self.bott(self.pool(c))
        z = self.c3(torch.cat([self.u3(z), c], 1))
        z = self.c2(torch.cat([self.u2(z), b], 1))
        z = self.c1(torch.cat([self.u1(z), a], 1))
        return self.out(z).squeeze(1)


