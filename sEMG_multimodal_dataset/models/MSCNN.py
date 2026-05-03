import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, window_size=300):
        super().__init__()
        W = window_size // 20
        kernel_sizes = [(1 * W, 3), (2 * W, 3), (3 * W, 3), (4 * W, 3), (5 * W, 3)]
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            pad = (ks[0] // 2, 1)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, base_filters, kernel_size=ks, padding=pad),
                nn.BatchNorm2d(base_filters),
                nn.ReLU(),
                nn.MaxPool2d((15, 1)),
                nn.Dropout2d(0.2),
            )
            self.branches.append(branch)

    def forward(self, x):
        return [branch(x) for branch in self.branches]


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        kh = 2 if x.shape[-2] >= 2 else 1
        kw = 2 if x.shape[-1] >= 2 else 1
        if kh > 1 or kw > 1:
            x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
        x = self.drop(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, num_classes=14, window_size=300, in_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = int(window_size)
        self.multi_scale = MultiScaleConvBlock(in_channels=1, base_filters=64, window_size=self.window_size)
        self.sep_blocks = nn.ModuleList([SeparableConvBlock(in_channels=64, out_channels=128) for _ in range(5)])
        self.conv1x1 = nn.Conv2d(5 * 128, 128, kernel_size=1)
        self.sep1 = SeparableConvBlock(128, 256)
        self.sep2 = SeparableConvBlock(256, 256)
        self.flatten = nn.Flatten()
        self.fc1 = None
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.transpose(1, 2).unsqueeze(1).contiguous()
        elif x.ndim != 4:
            raise ValueError(f"MSCNN expects input [B, 8, T] or [B, 1, L, C], got shape {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"MSCNN expects channel dim 1 after reshape, got shape {tuple(x.shape)}")
        if x.shape[-2:] != (self.window_size, self.in_channels):
            x = F.interpolate(x, size=(self.window_size, self.in_channels), mode="bilinear", align_corners=False)
        branches = self.multi_scale(x)
        branches_out = [self.sep_blocks[i](branches[i]) for i in range(5)]
        fused = torch.cat(branches_out, dim=1)
        fused = self.conv1x1(fused)
        feat = self.sep1(fused)
        feat = self.sep2(feat)
        feat = self.flatten(feat)
        if self.fc1 is None or self.fc1.in_features != feat.shape[1]:
            self.fc1 = nn.Linear(int(feat.shape[1]), 128).to(feat.device)
        feat = self.relu(self.fc1(feat))
        logits = self.fc2(feat)
        return self.softmax(logits)
