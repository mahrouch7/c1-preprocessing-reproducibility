"""Compact, explicitly adapted temporal ResNet, not a new algorithm claim."""
from __future__ import annotations

import torch
from torch import nn
from .io import require


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, dilation, eps, momentum):
        super().__init__()
        self.layers = nn.ModuleList()
        current = in_channels
        for kernel in kernels:
            require(kernel > 0 and kernel % 2 == 1, "Odd positive temporal kernels required")
            self.layers.append(nn.Sequential(
                nn.Conv1d(current, out_channels, kernel, padding=dilation * (kernel - 1) // 2,
                          dilation=dilation, bias=False),
                nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum)))
            current = out_channels
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
        return self.relu(x + shortcut)


class ResNet1D(nn.Module):
    def __init__(self, architecture, classes=12):
        super().__init__()
        widths, kernels, dilations = (architecture[k] for k in ("widths", "kernels", "block_dilations"))
        require(len(widths) == len(dilations) == 3 and len(kernels) == 3, "Exactly three three-convolution blocks required")
        require(all(w > 0 for w in widths) and all(d > 0 for d in dilations), "Invalid widths/dilations")
        blocks, current = [], 6
        for width, dilation in zip(widths, dilations):
            blocks.append(ResidualBlock(current, width, kernels, dilation,
                                        architecture["batch_norm_eps"], architecture["batch_norm_momentum"]))
            current = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(widths[-1], classes)
        self.receptive_field_samples = 1 + sum((k - 1) * d for d in dilations for k in kernels)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 6:
            raise ValueError("Expected batch-by-six-channels-by-time tensor")
        return self.classifier(self.pool(self.features(x)).squeeze(-1))

    def description(self, sampling_rate_hz=40000):
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "convolution_layers": sum(isinstance(m, nn.Conv1d) for m in self.modules()),
                "main_path_convolutions": 9, "receptive_field_samples": self.receptive_field_samples,
                "receptive_field_ms": 1000 * self.receptive_field_samples / sampling_rate_hz,
                "temporal_downsampling": False, "global_average_pooling": True}
