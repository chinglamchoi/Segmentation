import torch.nn.functional as F
import torch.nn as nn

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.bn2 = nn.BatchNorm2d(512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.bn3 = nn.BatchNorm2d(128)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.bn4 = nn.BatchNorm2d(64)
        self.outc = OutConv(64, n_classes)
        self.outc1 = nn.AdaptiveAvgPool2d((1,1))
        self.outc2 = nn.Linear(1,1)
        self.out_dec = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.bn1(self.down2(x2))
        x4 = self.down3(x3)
        x5 = self.bn2(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.bn3(self.up2(x, x3))
        x = self.up3(x, x2)
        x = self.bn4(self.up4(x, x1))
        logits = self.outc(x)
        out = self.outc1(logits)
        out = self.outc2(out)
        out = self.out_dec(out)
        return logits, out

def run_cnn():
    return UNet()
