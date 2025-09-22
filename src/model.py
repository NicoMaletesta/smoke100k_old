#!/usr/bin/env python3
# src/model.py
"""
UNetSegmenter adattato al progetto esistente (file originale fornito).
Modifiche richieste:
- backbone predefinito e unico: ResNet-50 (con pesi ImageNet se pretrained=True)
- attention: SEMPRE applicata (SE sul bottleneck, CBAM dopo up4)
- struttura del decoder mantenuta compatibile con il resto del codice:
  per evitare di cambiare signature/shape, i blocchi encoder di ResNet50
  vengono proiettati (1x1) alle dimensioni attese dal decoder originale.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# import attention dal file che hai richiesto (deve trovarsi in src/attention.py)
from src.attention import SEBlock, CBAM


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # initialize convs consistently
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[-2:] != skip.size()[-2:]:
            x = nn.functional.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSegmenter(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        """
        Mantengo la signature originale: (out_channels=1, pretrained=True)
        - backbone è forzato a ResNet50 (se pretrained=True usa ImageNet weights).
        - attention è sempre attiva (SE + CBAM).
        """
        super().__init__()

        # ResNet50 backbone (unico supportato)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # encoder pieces (conv1 .. layer4)
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1  # out channels: 256
        self.enc2 = backbone.layer2  # out channels: 512
        self.enc3 = backbone.layer3  # out channels: 1024
        self.enc4 = backbone.layer4  # out channels: 2048

        # Project encoder feature maps to sizes expected by original decoder:
        # enc4 -> 512, enc3 -> 256, enc2 -> 128, enc1 -> 64
        self.proj4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Bottleneck: dimensione ridotta a 512 per compatibilità con decoder esistente
        self.bottleneck = ConvBlock(512, 512)

        # Decoder (mantengo le stesse dimensioni logiche del progetto originale)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # Upsample finale per riportare alla risoluzione originale (da 256×256 → 512×512)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ATTENTION: sempre presente (SE sul bottleneck, CBAM dopo up4)
        self.attn_se = SEBlock(512)
        self.attn_cbam = CBAM(256)

        # final conv init
        nn.init.kaiming_normal_(self.final_conv.weight, nonlinearity='relu')
        if getattr(self.final_conv, "bias", None) is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)     # H/2 = 256
        x1 = self.maxpool(x0)    # H/4 = 128
        e1 = self.enc1(x1)       # channels 256
        e2 = self.enc2(e1)       # channels 512
        e3 = self.enc3(e2)       # channels 1024
        e4 = self.enc4(e3)       # channels 2048

        # Project encoder features to decoder-friendly channels
        p4 = self.proj4(e4)  # 512
        p3 = self.proj3(e3)  # 256
        p2 = self.proj2(e2)  # 128
        p1 = self.proj1(e1)  # 64

        # Bottleneck (operates on projected enc4)
        b = self.bottleneck(p4)
        b = self.attn_se(b)   # SE attention always applied

        # Decoder with skip connections (use projected skips)
        d4 = self.up4(b, p4)   # -> 256
        d4 = self.attn_cbam(d4)  # CBAM always applied here
        d3 = self.up3(d4, p3)   # -> 128
        d2 = self.up2(d3, p2)   # -> 64
        d1 = self.up1(d2, p1)   # -> 64
        d0 = self.up0(d1, x0)   # concat with x0 (64 channels)

        logits = self.final_conv(d0)            # [B, C, 256, 256]
        logits = self.final_upsample(logits)    # [B, C, 512, 512]

        return logits


def _test_model():
    print("Testing UNetSegmenter (ResNet-50 backbone) with final upsampling to 512×512...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNetSegmenter(out_channels=2, pretrained=False).to(device)

    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        logits = model(x)

    print("Output shape:", logits.shape)
    assert logits.shape == (1, 2, 512, 512), "Shape mismatch!"
    print("✅ Test passed!")


if __name__ == "__main__":
    _test_model()
