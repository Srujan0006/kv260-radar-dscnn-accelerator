"""DPU-friendly 2D CNN for 100x100 radar spectrograms."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Radar2DCNN(nn.Module):
    """Small 2D CNN intended to compile cleanly on DPUCZDX8G.

    Input: [N, 1, 100, 100]
    Output: [N, num_classes]
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
            nn.AvgPool2d(kernel_size=12),               # 12 -> 1
        )
        self.classifier = nn.Linear(c * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint_path: str, num_classes: int = 4, base_channels: int = 16, map_location="cpu") -> Radar2DCNN:
    """
    Load a locally generated training checkpoint.

    First try weights_only=True for newer PyTorch safety behavior.
    If that fails inside older/Vitis-AI PyTorch environments, fall back to
    normal torch.load because this checkpoint was generated locally by us.
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

    model = Radar2DCNN(
        num_classes=int(cfg.get("num_classes", num_classes)),
        base_channels=int(cfg.get("base_channels", base_channels)),
    )

    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model
