"""
HiFi-GAN vocoder utilities for inference.

This module provides a lightweight Generator implementation adapted from
https://github.com/jik876/hifi-gan (MIT License) so that we can load an
already-trained HiFi-GAN checkpoint and synthesize waveform samples from
log-Mel spectrograms predicted by the throat2air regression model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


@dataclass
class HiFiGANConfig:
    """Minimal subset of HiFi-GAN hyper parameters needed at inference time."""

    sampling_rate: int = 16000
    num_mels: int = 40
    upsample_rates: Sequence[int] = field(default_factory=lambda: (8, 8, 2, 2))
    upsample_kernel_sizes: Sequence[int] = field(default_factory=lambda: (16, 16, 4, 4))
    upsample_initial_channel: int = 512
    resblock: str = "1"
    resblock_kernel_sizes: Sequence[int] = field(default_factory=lambda: (3, 7, 11))
    resblock_dilation_sizes: Sequence[Sequence[int]] = field(
        default_factory=lambda: ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    )

    @staticmethod
    def from_dict(cfg: Dict) -> "HiFiGANConfig":
        base = HiFiGANConfig().__dict__.copy()
        # Merge only the fields we actually support; silently ignore extras from training configs.
        filtered = {k: v for k, v in cfg.items() if k in base}
        base.update(filtered)
        return HiFiGANConfig(**base)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Sequence[int] = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=_get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, cfg: HiFiGANConfig):
        super().__init__()
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)

        resblock_cls = ResBlock1 if cfg.resblock == "1" else ResBlock2

        self.conv_pre = weight_norm(nn.Conv1d(cfg.num_mels, cfg.upsample_initial_channel, kernel_size=7, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        in_channels = cfg.upsample_initial_channel
        for i in range(self.num_upsamples):
            out_channels = in_channels // 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        cfg.upsample_kernel_sizes[i],
                        cfg.upsample_rates[i],
                        padding=(cfg.upsample_kernel_sizes[i] - cfg.upsample_rates[i]) // 2,
                    )
                )
            )
            for j in range(self.num_kernels):
                self.resblocks.append(
                    resblock_cls(out_channels, cfg.resblock_kernel_sizes[j], cfg.resblock_dilation_sizes[j])
                )
            in_channels = out_channels

        self.conv_post = weight_norm(nn.Conv1d(in_channels, 1, kernel_size=7, padding=3))
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            res_sum = 0
            for j in range(self.num_kernels):
                res_sum += self.resblocks[i * self.num_kernels + j](x)
            x = res_sum / self.num_kernels

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class HiFiGANVocoder:
    """Wrapper around the HiFi-GAN generator for inference-time usage."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        cfg = HiFiGANConfig()
        if config_path:
            with open(config_path, "r", encoding="utf-8") as cfg_file:
                user_cfg = json.load(cfg_file)
            cfg = HiFiGANConfig.from_dict(user_cfg)
        self.cfg = cfg

        self.generator = Generator(cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        if "generator" in state:
            state = state["generator"]
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.generator.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys in HiFi-GAN checkpoint: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in HiFi-GAN checkpoint: {sorted(unexpected)}")
        self.generator.remove_weight_norm()
        self.generator.eval()

    def mel_to_audio(self, log_mel: np.ndarray) -> np.ndarray:
        """Convert a [T, num_mels] log-Mel array to waveform samples."""
        if log_mel.ndim != 2:
            raise ValueError(f"log_mel must be 2D, got shape {log_mel.shape}")
        if log_mel.shape[1] != self.cfg.num_mels:
            raise ValueError(
                f"Mel dimension mismatch: model expects {self.cfg.num_mels}, but received {log_mel.shape[1]}"
            )

        mel = torch.from_numpy(log_mel.T).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            audio = self.generator(mel)
        return audio.squeeze().cpu().numpy().astype(np.float32)

    def __call__(self, log_mel: np.ndarray) -> np.ndarray:
        return self.mel_to_audio(log_mel)


def load_hifigan_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as cfg_file:
        return json.load(cfg_file)


__all__ = [
    "HiFiGANVocoder",
    "HiFiGANConfig",
]
