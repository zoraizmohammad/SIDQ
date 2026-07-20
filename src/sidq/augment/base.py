"""Base augmentation interface with metadata and determinism."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TransformMetadata:
    """Metadata recorded for each transform application."""

    transform_name: str
    parameters: dict = field(default_factory=dict)
    seed_used: int = 0
    duration_change: float = 0.0
    requires_external: list[str] = field(default_factory=list)


class AudioTransform(ABC):
    """Base class for all audio augmentation transforms."""

    def __init__(
        self,
        probability: float = 1.0,
        seed: int | None = None,
    ):
        self.probability = probability
        self._base_seed = seed
        self._call_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Transform name for logging and metadata."""
        ...

    @property
    def external_requirements(self) -> list[str]:
        """List of external binaries required (e.g., ['ffmpeg'])."""
        return []

    @property
    def is_differentiable(self) -> bool:
        """Whether this transform supports gradient computation."""
        return False

    def _get_rng(self) -> np.random.Generator:
        """Get deterministic RNG for this call."""
        seed = self._base_seed + self._call_count if self._base_seed is not None else None
        self._call_count += 1
        return np.random.default_rng(seed)

    def __call__(
        self, waveform: torch.Tensor, sample_rate: int = 16000
    ) -> tuple[torch.Tensor, TransformMetadata]:
        """Apply transform with probability check.

        Returns:
            Tuple of (transformed_waveform, metadata).
        """
        rng = self._get_rng()

        if rng.random() > self.probability:
            return waveform, TransformMetadata(
                transform_name=f"{self.name}:skipped",
                seed_used=self._base_seed or 0,
            )

        result, metadata = self.apply(waveform, sample_rate, rng)

        if not torch.isfinite(result).all():
            return waveform, TransformMetadata(
                transform_name=f"{self.name}:failed_nonfinite",
                seed_used=self._base_seed or 0,
            )

        return result, metadata

    @abstractmethod
    def apply(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, TransformMetadata]:
        """Apply the transform (called after probability check)."""
        ...

    def is_available(self) -> bool:
        """Check if all external requirements are available."""
        import shutil

        return all(shutil.which(req) for req in self.external_requirements)
