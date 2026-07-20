"""Raw-waveform specialist model for corruption detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """Sinc-based convolution layer (simplified parameterized filterbank)."""

    def __init__(self, out_channels: int = 70, kernel_size: int = 251, sample_rate: int = 16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize center frequencies linearly in mel scale
        low_hz = 30.0
        high_hz = sample_rate / 2 - (sample_rate / 2 - low_hz) / (out_channels + 1)
        hz = torch.linspace(low_hz, high_hz, out_channels)
        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))

        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * torch.pi * torch.arange(-n, 0) / sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sinc filterbank.

        Args:
            x: (batch, 1, samples) or (batch, samples)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        low = self.low_hz_.abs() + 1.0
        high = (low + self.band_hz_.abs()).clamp(max=self.sample_rate / 2)

        low_pass1 = 2 * low / self.sample_rate * torch.sinc(2 * low * self.n_ / (2 * torch.pi))
        low_pass2 = 2 * high / self.sample_rate * torch.sinc(2 * high * self.n_ / (2 * torch.pi))

        band_pass = low_pass2 - low_pass1
        band_pass = torch.cat([band_pass, torch.zeros(band_pass.shape[0], 1, device=x.device), band_pass.flip(dims=[1])], dim=1)

        # Window
        window = torch.hamming_window(self.kernel_size, device=x.device)
        filters = band_pass * window

        return F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RawSpecialist(nn.Module):
    """Raw waveform specialist for corruption-robust detection.

    Processes raw audio directly without SSL features.
    Designed to complement SSL-based models on heavily corrupted audio.
    """

    def __init__(
        self,
        sinc_channels: int = 70,
        channels: list[int] | None = None,
        num_classes: int = 2,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 128, 256, 256]

        self.sinc = SincConv(out_channels=sinc_channels)

        layers: list[nn.Module] = []
        in_ch = sinc_channels - 1  # SincConv uses n-1 bandpass filters
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.3),
                ResBlock(out_ch),
            ])
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass from raw waveform to logits.

        Args:
            waveform: (batch, samples) raw audio.
            lengths: unused, kept for API compatibility.

        Returns:
            Logits (batch, num_classes).
        """
        x = self.sinc(waveform)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
