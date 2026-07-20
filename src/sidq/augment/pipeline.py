"""Augmentation pipeline composing multiple transforms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sidq.augment.base import AudioTransform, TransformMetadata


@dataclass
class PipelineResult:
    """Result from an augmentation pipeline application."""

    waveform: torch.Tensor
    applied_transforms: list[TransformMetadata] = field(default_factory=list)
    num_applied: int = 0


class AugmentationPipeline:
    """Compose multiple transforms with configurable selection."""

    def __init__(
        self,
        transforms: list[AudioTransform],
        max_transforms: int = 3,
        min_transforms: int = 0,
        clean_probability: float = 0.3,
        seed: int | None = None,
    ):
        self.transforms = transforms
        self.max_transforms = max_transforms
        self.min_transforms = min_transforms
        self.clean_probability = clean_probability
        self._base_seed = seed
        self._call_count = 0

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> PipelineResult:
        """Apply augmentation pipeline to a waveform."""
        if self._base_seed is not None:
            rng = np.random.default_rng(self._base_seed + self._call_count)
        else:
            rng = np.random.default_rng()
        self._call_count += 1

        if rng.random() < self.clean_probability:
            return PipelineResult(waveform=waveform, num_applied=0)

        available = [t for t in self.transforms if t.is_available()]
        if not available:
            return PipelineResult(waveform=waveform, num_applied=0)

        n_transforms = rng.integers(
            self.min_transforms, min(self.max_transforms, len(available)) + 1
        )

        if n_transforms == 0:
            return PipelineResult(waveform=waveform, num_applied=0)

        selected_indices = rng.choice(len(available), size=n_transforms, replace=False)
        selected = [available[i] for i in selected_indices]

        current = waveform
        applied: list[TransformMetadata] = []

        for transform in selected:
            current, meta = transform(current, sample_rate)
            applied.append(meta)

        return PipelineResult(
            waveform=current,
            applied_transforms=applied,
            num_applied=len(applied),
        )
