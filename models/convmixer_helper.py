#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch.nn as nn
import torch.nn.functional as F


class ConvMixerPatchEmbed(nn.Module):
    """ConvMixer 2D PatchEmbeding."""

    def __init__(
        self, dim_in=3, dim_out=768, kernel=7, stride=7, padding=0, activation=nn.GELU
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding
        )
        self.act = activation()
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)

        return x


class ConvMixerHead(nn.Module):
    """ConvMixer Head"""

    def __init__(self, dim=768, dropout_rate=0.0, classes=400):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim, classes, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.fc(x)

        return x


class WSConvMixerBlock(nn.Module):
    """Main block of WSConvMixer"""

    def __init__(self, dim=768, kernel_size=7, padding=1, activation=nn.GELU):
        super().__init__()
        self.dwise_blk = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=padding),
            activation(),
            nn.BatchNorm2d(dim),
        )
        self.act2 = activation()
        self.bn2 = nn.BatchNorm2d(dim)

    def pwise_blk(self, x, pwise_w, pwise_b):
        x = F.conv2d(x, pwise_w, pwise_b)
        x = self.act2(x)
        x = self.bn2(x)
        return x

    def forward(self, x, pwise_w, pwise_b):
        x = x + self.dwise_blk(x)
        x = x + self.pwise_blk(x, pwise_w, pwise_b)
        return x
