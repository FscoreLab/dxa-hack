"""U-Net с предобученным энкодером resnet34 (ImageNet): на малой выборке сеть с нуля путает уровни."""
import torch
import torchvision.models as M
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))


class UNetR34(nn.Module):
    def __init__(self, n_cls: int = 5, pretrained: bool = True):
        super().__init__()
        w = M.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = M.resnet34(weights=w)
        # Вход одноканальный: веса первой свёртки суммируются по каналам RGB.
        c1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            c1.weight.copy_(r.conv1.weight.sum(1, keepdim=True))
        self.stem = nn.Sequential(c1, r.bn1, r.relu)      # /2,  64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.u4, self.c4 = nn.Upsample(scale_factor=2, mode="bilinear"), _block(512 + 256, 256)
        self.u3, self.c3 = nn.Upsample(scale_factor=2, mode="bilinear"), _block(256 + 128, 128)
        self.u2, self.c2 = nn.Upsample(scale_factor=2, mode="bilinear"), _block(128 + 64, 64)
        self.u1, self.c1_ = nn.Upsample(scale_factor=2, mode="bilinear"), _block(64 + 64, 48)
        self.u0 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out = nn.Sequential(_block(48, 32), nn.Conv2d(32, n_cls, 1))

    def forward(self, x):
        s = self.stem(x)               # /2
        a = self.l1(self.pool(s))      # /4
        b = self.l2(a)                 # /8
        c = self.l3(b)                 # /16
        d = self.l4(c)                 # /32
        z = self.c4(torch.cat([self.u4(d), c], 1))
        z = self.c3(torch.cat([self.u3(z), b], 1))
        z = self.c2(torch.cat([self.u2(z), a], 1))
        z = self.c1_(torch.cat([self.u1(z), s], 1))
        return self.out(self.u0(z))
