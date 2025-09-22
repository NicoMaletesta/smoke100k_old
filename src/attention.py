#!/usr/bin/env python3
# src/attention.py
"""
Production-ready SEBlock e CBAM usati dal progetto.
Nessuna dipendenza esterna (cv2/matplotlib): solo PyTorch.
Le classi espongono buffer leggeri last_weights / last_spatial utili per debugging.
"""

from typing import Optional
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation con conv1x1 per MLP, dropout opzionale e residual skip."""
    def __init__(self, channels: int, reduction: int = 16,
                 temp: float = 0.5, drop_p: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        mid = max(min(channels // reduction, 64), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.temp = float(temp) if temp > 0 else 1.0
        self.dropout = nn.Dropout2d(p=drop_p) if (drop_p and drop_p > 0.0) else nn.Identity()

        self.register_buffer("last_weights", torch.zeros(channels), persistent=False)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last) if x.is_contiguous() else x
        y = self.avg_pool(x)      # [B,C,1,1]
        y = self.fc(y)            # [B,C,1,1]
        y = self.sigmoid(y)
        if self.temp != 1.0:
            y = y.pow(1.0 / self.temp)
        with torch.no_grad():
            try:
                self.last_weights = y[0].view(-1).detach().cpu()
            except Exception:
                pass
        out = x * y.expand_as(x)
        out = self.dropout(out)
        return x + out  # residual skip

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class CBAM(nn.Module):
    """Channel + Spatial attention (multi-scale spatial conv), con residual skips."""
    def __init__(self, channels: int, reduction: int = 16,
                 kernel_size: int = 7, drop_p: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        mid = max(min(channels // reduction, 64), 4)

        # Channel attention
        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)
        self.max_pool_c = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid_c = nn.Sigmoid()

        # Spatial attention (3x3 + kxk)
        pad_k = kernel_size // 2
        self.conv_sp3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_spk = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad_k, bias=False)
        self.sigmoid_s = nn.Sigmoid()

        self.dropout = nn.Dropout2d(p=drop_p) if (drop_p and drop_p > 0.0) else nn.Identity()
        self.register_buffer("last_spatial", torch.zeros(1, 1, 1, 1), persistent=False)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last) if x.is_contiguous() else x

        # Channel attention
        avg = self.avg_pool_c(x)
        mx = self.max_pool_c(x)
        ca = self.mlp(avg) + self.mlp(mx)
        ca = self.sigmoid_c(ca)
        x_ca = x * ca.expand_as(x)
        ca_out = x + self.dropout(x_ca)

        # Spatial attention
        avg_map = ca_out.mean(dim=1, keepdim=True)
        max_map, _ = ca_out.max(dim=1, keepdim=True)
        sa_in = torch.cat([avg_map, max_map], dim=1)
        s3 = self.conv_sp3(sa_in)
        sk = self.conv_spk(sa_in)
        sa = self.sigmoid_s(s3 + sk)
        with torch.no_grad():
            try:
                self.last_spatial = sa[0:1].detach().cpu()
            except Exception:
                pass
        sa_out = ca_out * sa.expand_as(ca_out)
        return ca_out + self.dropout(sa_out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# quick smoke test (esegue solo se chiamato come script)
def _quick_test():
    x = torch.randn(2, 64, 32, 32)
    se = SEBlock(64)
    cbam = CBAM(64)
    y1 = se(x)
    y2 = cbam(x)
    print("SE out:", y1.shape, "CBAM out:", y2.shape)


if __name__ == "__main__":
    _quick_test()
