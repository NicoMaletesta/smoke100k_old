import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


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
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.bottleneck = ConvBlock(512, 512)

        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Upsample finale per riportare alla risoluzione originale (da 256×256 → 512×512)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0 = self.initial(x)     # H/2 = 256
        x1 = self.maxpool(x0)    # H/4 = 128
        e1 = self.enc1(x1)       # 128
        e2 = self.enc2(e1)       # 64
        e3 = self.enc3(e2)       # 32
        e4 = self.enc4(e3)       # 16

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)     # 32
        d3 = self.up3(d4, e3)    # 64
        d2 = self.up2(d3, e2)    # 128
        d1 = self.up1(d2, e1)    # 256
        d0 = self.up0(d1, x0)    # ancora 256

        logits = self.final_conv(d0)            # [B, C, 256, 256]
        logits = self.final_upsample(logits)    # [B, C, 512, 512]

        return logits


def _test_model():
    print("Testing UNetSegmenter with final upsampling to 512×512...")
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
