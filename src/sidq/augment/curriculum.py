"""Corruption curriculum: staged augmentation profiles."""

from __future__ import annotations

from sidq.augment.base import AudioTransform
from sidq.augment.codecs import CodecTransform
from sidq.augment.laundering import LaunderingTransform
from sidq.augment.noise import AdditiveNoiseTransform
from sidq.augment.pipeline import AugmentationPipeline
from sidq.augment.rawboost import RawBoostTransform
from sidq.augment.rerecording import RerecordingTransform
from sidq.augment.reverberation import ReverbTransform
from sidq.augment.telephony import TelephonyTransform


def build_pipeline(
    profile: str = "none",
    seed: int = 42,
) -> AugmentationPipeline:
    """Build augmentation pipeline from named profile.

    Profiles:
        none: No augmentation (clean only)
        mild: 60-70% clean, 30-40% one mild corruption
        robust: 25-40% clean, one-two realistic corruptions
        hard: 10-25% clean, one-three including severe channel cases
    """
    if profile == "none":
        return AugmentationPipeline(
            transforms=[],
            clean_probability=1.0,
            seed=seed,
        )
    elif profile == "mild":
        transforms = _mild_transforms(seed)
        return AugmentationPipeline(
            transforms=transforms,
            clean_probability=0.65,
            min_transforms=1,
            max_transforms=1,
            seed=seed,
        )
    elif profile == "robust":
        transforms = _robust_transforms(seed)
        return AugmentationPipeline(
            transforms=transforms,
            clean_probability=0.30,
            min_transforms=1,
            max_transforms=2,
            seed=seed,
        )
    elif profile == "hard":
        transforms = _hard_transforms(seed)
        return AugmentationPipeline(
            transforms=transforms,
            clean_probability=0.15,
            min_transforms=1,
            max_transforms=3,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown augmentation profile: {profile}")


def _mild_transforms(seed: int) -> list[AudioTransform]:
    """Mild corruptions: gentle codec, low noise, short reverb."""
    return [
        AdditiveNoiseTransform(
            noise_types=["white", "pink"],
            snr_range=(25.0, 40.0),
            seed=seed + 1,
        ),
        ReverbTransform(
            room_profiles=["small"],
            wet_ratio_range=(0.1, 0.3),
            seed=seed + 2,
        ),
        CodecTransform(
            codecs=["mp3_128k", "aac_128k", "opus_64k"],
            seed=seed + 3,
        ),
    ]


def _robust_transforms(seed: int) -> list[AudioTransform]:
    """Robust corruptions: realistic codec, noise, reverb, telephony."""
    return [
        AdditiveNoiseTransform(
            noise_types=["white", "pink", "brown"],
            snr_range=(10.0, 30.0),
            seed=seed + 10,
        ),
        ReverbTransform(
            room_profiles=["small", "medium"],
            wet_ratio_range=(0.2, 0.6),
            seed=seed + 11,
        ),
        CodecTransform(
            codecs=["mp3_64k", "mp3_128k", "aac_64k", "opus_32k", "ogg_64k"],
            seed=seed + 12,
        ),
        TelephonyTransform(mode="random", seed=seed + 13),
        RawBoostTransform(
            modes=["linear_convolutive", "stationary"],
            seed=seed + 14,
        ),
    ]


def _hard_transforms(seed: int) -> list[AudioTransform]:
    """Hard corruptions: severe codec chains, heavy noise, full channel."""
    return [
        AdditiveNoiseTransform(
            noise_types=["white", "pink", "brown"],
            snr_range=(5.0, 20.0),
            seed=seed + 20,
        ),
        ReverbTransform(
            room_profiles=["medium", "large"],
            wet_ratio_range=(0.4, 0.8),
            seed=seed + 21,
        ),
        CodecTransform(
            codecs=["mp3_64k", "aac_64k", "opus_32k", "g711_ulaw", "g711_alaw"],
            double_encode=True,
            seed=seed + 22,
        ),
        TelephonyTransform(mode="narrowband", seed=seed + 23),
        RerecordingTransform(seed=seed + 24),
        LaunderingTransform(seed=seed + 25),
        RawBoostTransform(
            modes=["combined", "impulsive"],
            seed=seed + 26,
        ),
    ]
