"""Typed configuration schemas for SIDQ-T2."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED


class Label(int, Enum):
    """Audio classification labels."""

    BONAFIDE = LABEL_BONAFIDE
    SPOOFED = LABEL_SPOOFED


class AugmentationProfile(str, Enum):
    """Augmentation severity profiles."""

    NONE = "none"
    MILD = "mild"
    ROBUST = "robust"
    HARD = "hard"


class ModelBackend(str, Enum):
    """Backend classifier type."""

    AASIST = "aasist"
    ECAPA = "ecapa"
    ATTENTIVE = "attentive_pooling"


class SSLFrontend(str, Enum):
    """Self-supervised learning frontend."""

    XLSR_300M = "xlsr_300m"
    WAVLM_BASE_PLUS = "wavlm_base_plus"
    WAVLM_LARGE = "wavlm_large"


class AudioConfig(BaseModel):
    """Audio preprocessing configuration."""

    sample_rate: int = 16_000
    mono: bool = True
    max_duration_sec: float = 20.0
    min_duration_sec: float = 0.5
    crop_duration_sec: float = 4.0
    normalize: bool = True

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in (8000, 16000, 22050, 44100, 48000):
            raise ValueError(f"Unsupported sample rate: {v}")
        return v


class DataConfig(BaseModel):
    """Dataset loading configuration."""

    data_root: Path | None = None
    streaming: bool = False
    manifest_path: Path | None = None
    split: str = "train"
    num_workers: int = 4
    prefetch_factor: int = 2


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    frontend: SSLFrontend = SSLFrontend.XLSR_300M
    backend: ModelBackend = ModelBackend.AASIST
    frontend_checkpoint: str | None = None
    freeze_frontend: bool = True
    layer_weights: bool = False
    hidden_dim: int = 160
    num_classes: int = 2


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    frontend_lr: float | None = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accumulation: int = 1
    mixed_precision: bool = True
    seed: int = 42
    augmentation: AugmentationProfile = AugmentationProfile.NONE
    early_stopping_patience: int = 5


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    multi_crop: bool = False
    crop_overlap: float = 0.5
    aggregation: str = "mean"
    compute_bootstrap: bool = False
    bootstrap_n: int = 1000

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        allowed = {"mean", "median", "logsumexp"}
        if v not in allowed:
            raise ValueError(f"aggregation must be one of {allowed}")
        return v


class FusionConfig(BaseModel):
    """Score fusion configuration."""

    method: str = "weighted_mean"
    normalize: str = "zscore"
    models: list[str] = Field(default_factory=list)
    weights: list[float] | None = None

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"uniform", "weighted_mean", "logistic", "rank"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str
    audio: AudioConfig = Field(default_factory=AudioConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: Path = Path("artifacts/experiments")


class SubmissionRecord(BaseModel):
    """Metadata for a competition submission."""

    audio_id: str
    logit: float

    @field_validator("logit")
    @classmethod
    def validate_logit(cls, v: float) -> float:
        import math

        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"logit must be finite, got {v}")
        return v
