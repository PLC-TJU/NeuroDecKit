
"""
modiified from https://github.com/youshuoji/HCANN/blob/main/main.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from einops import rearrange

from .base import SkorchNet2


@SkorchNet2
class HCANN(nn.Module):
    def __init__(self, fs, channels, dims, depth, heads,  num_classes, dim_initial, dropout=0.2, ):
        super(HCANN, self).__init__()
        self.dim_initial = dim_initial
        self.conv_T_1_1x16 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 16), stride=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_2_1x16 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 16), stride=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_3_1x8 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            SeparableConv2d(in_channels=8, out_channels=16, kernel_size=(1, 8), stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_4_1x8 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            SeparableConv2d(in_channels=16, out_channels=16, kernel_size=(1, 8), stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )

        self.conv_S = nn.Sequential(
            Conv2dWithConstraint(in_channels=16, out_channels=16, kernel_size=(64, 1), bias=False, stride=(64, 1)),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.Hardswish()
        )

        self.AvgPooling1 = nn.AdaptiveAvgPool2d((64, self.dim_initial//4))
        self.AvgPooling2 = nn.AdaptiveAvgPool2d((64, self.dim_initial//32))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([])
        self.transformer.append(transformer(depth=depth[0], dim=dims[0], heads=heads[0], hidden_feed=2, channel=8, dropout=dropout))
        self.transformer.append(transformer(depth=depth[1], dim=dims[1], heads=heads[1], hidden_feed=4, channel=16, dropout=dropout))


        self.fc = nn.Linear(dim_initial*16//32, 8)

    def forward(self, x):
        out = x
        # out = self.conv_channel64(x)
        out = rearrange(out, 'b c t -> b 1 c t')
        out = self.conv_T_1_1x16(out)
        out = self.conv_T_2_1x16(out)
        out = self.AvgPooling1(out)
        out = self.dropout(out)

        out = self.transformer[0](out)

        out = self.conv_T_3_1x8(out)
        out = self.conv_T_4_1x8(out)


        out = self.AvgPooling2(out)
        out = self.dropout(out)
        out = self.transformer[1](out)

        out = self.conv_S(out)


        out = out.flatten(start_dim=1)
        # out = self.fc2(self.fc(out))
        out = self.fc(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, channel, dropout=0.0):
        super().__init__()

        self.feed_net = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 2)),
            nn.Conv2d(in_channels=channel, out_channels=channel*hidden, kernel_size=(5, 1), stride=1, bias=False),
            nn.BatchNorm2d(channel*hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=channel*hidden, out_channels=channel, kernel_size=1, stride=1, bias=False),

            nn.BatchNorm2d(channel),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed_net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.heads = heads
        assert dim % heads == 0
        self.dim_head = dim // self.heads

        self.scale = self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, heads * self.dim_head * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(heads * self.dim_head, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class transformer(nn.Module):
    def __init__(self, depth, dim, heads, hidden_feed, channel, dropout=0.):
        super(transformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout)),
                Position_wise_Feed_Forward(dim, hidden_feed, channel, dropout)
            ]))

    def forward(self, x):
        for attn, feed in self.layers:
            x = attn(x) + x
            # x = attn(x)
            x = feed(x) + x
        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

