"""
unet.py — Modular U-Net for binary/multi-class medical image segmentation.

Architecture:
  Encoder: N depth levels, each = DoubleConvBlock + MaxPool
  Bottleneck: DoubleConvBlock
  Decoder: N levels, each = Upsample + Concat(skip) + DoubleConvBlock
  Head: 1x1 conv → logits  (sigmoid/softmax applied in loss or metric)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(inplace=True),
            "leaky": nn.LeakyReLU(0.1, inplace=True),
            "elu": nn.ELU(inplace=True)}.get(name, nn.ReLU(inplace=True))


def _norm(name: str, channels: int) -> nn.Module:
    return {"batch": nn.BatchNorm2d(channels),
            "instance": nn.InstanceNorm2d(channels),
            "none": nn.Identity()}.get(name, nn.BatchNorm2d(channels))


class DoubleConvBlock(nn.Module):
    """Conv→Norm→Act repeated twice, optional dropout after second."""

    def __init__(self, in_ch: int, out_ch: int,
                 act: str = "relu", norm: str = "batch",
                 dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=(norm == "none")),
            _norm(norm, out_ch),
            _activation(act),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=(norm == "none")),
            _norm(norm, out_ch),
            _activation(act),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConvBlock + MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.conv = DoubleConvBlock(in_ch, out_ch, **kwargs)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Upsample → concat skip → DoubleConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 bilinear: bool = True, **kwargs):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, 1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConvBlock(in_ch // 2 + skip_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip if needed (handles odd spatial dims)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Configurable U-Net.

    Args:
        in_channels:   input image channels (1 or 3)
        num_classes:   1 for binary (logits → sigmoid), >1 for multi-class
        depth:         number of encoder/decoder levels (default 4)
        init_filters:  base number of feature maps (doubled each level)
        dropout:       dropout probability in conv blocks
        activation:    'relu' | 'leaky' | 'elu'
        norm:          'batch' | 'instance' | 'none'
        bilinear:      True → bilinear upsample; False → transposed conv
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        depth: int = 4,
        init_filters: int = 32,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str = "batch",
        bilinear: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes

        conv_kwargs = dict(act=activation, norm=norm, dropout=dropout)

        # Encoder
        self.encoders: nn.ModuleList = nn.ModuleList()
        ch = in_channels
        enc_channels: List[int] = []
        for d in range(depth):
            out_ch = init_filters * (2 ** d)
            self.encoders.append(EncoderBlock(ch, out_ch, **conv_kwargs))
            enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        bottleneck_ch = init_filters * (2 ** depth)
        self.bottleneck = DoubleConvBlock(ch, bottleneck_ch, **conv_kwargs)
        ch = bottleneck_ch

        # Decoder (mirror encoder in reverse)
        self.decoders: nn.ModuleList = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = enc_channels[d]
            out_ch  = init_filters * (2 ** d)
            self.decoders.append(
                DecoderBlock(ch, skip_ch, out_ch,
                             bilinear=bilinear, **conv_kwargs)
            )
            ch = out_ch

        # Output head
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path — collect skips
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path — use skips in reverse
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)   # raw logits


def build_unet(cfg: dict) -> UNet:
    """Convenience factory from config dict."""
    u = cfg.get("unet", {})
    d = cfg.get("dataset", {})
    return UNet(
        in_channels  = d.get("in_channels", 3),
        num_classes  = d.get("num_classes", 1),
        depth        = u.get("depth", 4),
        init_filters = u.get("init_filters", 32),
        dropout      = u.get("dropout", 0.2),
        activation   = u.get("activation", "relu"),
        norm         = u.get("norm", "batch"),
        bilinear     = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / \
               (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal).mean()


class CombinedLoss(nn.Module):
    """BCE/Focal + Dice combined loss."""

    def __init__(self, mode: str = "bce_dice",
                 bce_w: float = 0.5, dice_w: float = 0.5):
        super().__init__()
        assert abs(bce_w + dice_w - 1.0) < 1e-5, "weights must sum to 1"
        self.mode   = mode
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.dice   = DiceLoss()
        if mode == "focal_dice":
            self.base = FocalLoss()
        else:
            self.base = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.base(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


def build_loss(cfg: dict) -> nn.Module:
    t = cfg["training"]
    return CombinedLoss(
        mode   = t.get("loss", "bce_dice"),
        bce_w  = t.get("bce_weight", 0.5),
        dice_w = t.get("dice_weight", 0.5),
    )
