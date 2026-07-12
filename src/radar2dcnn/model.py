"""DPU-friendly 2D CNN for 100x100 radar spectrograms.

This version avoids Linear/MatMul and uses a 1x1 Conv2d classifier so that
Vitis AI DPUCZDX8G has a better chance of mapping the graph to DPU.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Radar2DCNN(nn.Module):
    """DPU-friendly CNN.

    Input:
        [N, 1, 100, 100]

    Output:
        [N, num_classes]
    """

    def __init__(self, num_classes: int = 4, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.features = nn.Sequential(
            ConvBNReLU(1, c),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 100 -> 50

            ConvBNReLU(c, c * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 50 -> 25

            ConvBNReLU(c * 2, c * 4),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 25 -> 12

            ConvBNReLU(c * 4, c * 4),
            nn.AvgPool2d(kernel_size=12, stride=12),    # 12 -> 1
        )

        # DPU-friendly replacement for Linear(c*4, num_classes).
        self.classifier_conv = nn.Conv2d(
            c * 4,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier_conv(x)     # [N, num_classes, 1, 1]
        x = torch.flatten(x, 1)         # [N, num_classes]
        return x


class DWConvBlock(nn.Module):
    """Depthwise 3x3 plus pointwise 1x1 convolution block."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RadarDSCNN2D(nn.Module):
    """DPU-friendly depthwise-separable CNN for 1x100x100 inputs."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DWConvBlock(16, 32, pool=True),
            DWConvBlock(32, 64, pool=True),
            DWConvBlock(64, 96, pool=True),
            DWConvBlock(96, 128, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def build_model(
    architecture: str,
    num_classes: int = 4,
    base_channels: int = 16,
) -> nn.Module:
    """Build one of the controlled-experiment model architectures."""
    if architecture == "radar2dcnn":
        return Radar2DCNN(num_classes=num_classes, base_channels=base_channels)
    if architecture == "radar_dscnn2d":
        return RadarDSCNN2D(num_classes=num_classes)
    raise ValueError(f"Unsupported model architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 4,
    base_channels: int = 16,
    map_location="cpu",
) -> nn.Module:
    """Load a locally generated checkpoint.

    Tries weights_only=True first. Falls back for older/Vitis-AI PyTorch.
    """
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        ckpt = torch.load(
            checkpoint_path,
            map_location=map_location,
        )
    except Exception as e:
        print("[INFO] weights_only=True checkpoint load failed.")
        print("[INFO] Falling back to weights_only=False for trusted local checkpoint.")
        print("[INFO] Original error:", repr(e))
        ckpt = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

    cfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}

    model = build_model(
        architecture=str(cfg.get("architecture", "radar2dcnn")),
        num_classes=int(cfg.get("num_classes", num_classes)),
        base_channels=int(cfg.get("base_channels", base_channels)),
    )

    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model
