"""Deterministic corruption bank for validation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sidq.augment.codecs import CodecTransform
from sidq.augment.noise import AdditiveNoiseTransform
from sidq.augment.rawboost import RawBoostTransform
from sidq.augment.rerecording import RerecordingTransform
from sidq.augment.reverberation import ReverbTransform
from sidq.augment.telephony import TelephonyTransform


@dataclass
class CorruptionCondition:
    """A named corruption condition for evaluation."""

    name: str
    family: str
    severity: str
    transform_fn: callable
    seed: int = 0


@dataclass
class CorruptionBankResult:
    """Results from evaluating across a corruption bank."""

    clean_eer: float
    per_condition: dict[str, float] = field(default_factory=dict)
    per_family: dict[str, float] = field(default_factory=dict)
    worst_condition: str = ""
    worst_eer: float = 0.0
    mean_eer: float = 0.0


def build_corruption_bank(seed: int = 42) -> list[CorruptionCondition]:
    """Build a deterministic set of corruption conditions for evaluation."""
    conditions: list[CorruptionCondition] = []

    # Codec conditions
    for codec_name in ["mp3_64k", "mp3_128k", "opus_32k", "g711_ulaw"]:
        conditions.append(CorruptionCondition(
            name=f"codec_{codec_name}",
            family="codec",
            severity="medium",
            transform_fn=lambda s=seed, c=codec_name: CodecTransform(codecs=[c], seed=s),
            seed=seed,
        ))

    # Noise conditions
    for snr in [5, 10, 20]:
        conditions.append(CorruptionCondition(
            name=f"noise_white_snr{snr}",
            family="noise",
            severity="low" if snr >= 20 else ("medium" if snr >= 10 else "high"),
            transform_fn=lambda s=seed, snr_v=snr: AdditiveNoiseTransform(
                noise_types=["white"], snr_range=(snr_v, snr_v), seed=s
            ),
            seed=seed + snr,
        ))

    # Reverberation conditions
    for room in ["small", "medium", "large"]:
        conditions.append(CorruptionCondition(
            name=f"reverb_{room}",
            family="reverberation",
            severity="low" if room == "small" else ("medium" if room == "medium" else "high"),
            transform_fn=lambda s=seed, r=room: ReverbTransform(room_profiles=[r], seed=s),
            seed=seed,
        ))

    # Telephony
    conditions.append(CorruptionCondition(
        name="telephony_narrowband",
        family="telephony",
        severity="high",
        transform_fn=lambda s=seed: TelephonyTransform(mode="narrowband", seed=s),
        seed=seed,
    ))

    # RawBoost
    conditions.append(CorruptionCondition(
        name="rawboost_combined",
        family="rawboost",
        severity="medium",
        transform_fn=lambda s=seed: RawBoostTransform(modes=["combined"], seed=s),
        seed=seed,
    ))

    # Re-recording
    conditions.append(CorruptionCondition(
        name="rerecording",
        family="rerecording",
        severity="high",
        transform_fn=lambda s=seed: RerecordingTransform(seed=s),
        seed=seed,
    ))

    return conditions


def apply_corruption_bank(
    waveforms: list[torch.Tensor],
    conditions: list[CorruptionCondition],
    sample_rate: int = 16000,
) -> dict[str, list[torch.Tensor]]:
    """Apply all conditions to a set of waveforms.

    Returns dict mapping condition_name -> list of corrupted waveforms.
    """
    results: dict[str, list[torch.Tensor]] = {}

    for condition in conditions:
        transform = condition.transform_fn()
        corrupted = []
        for wav in waveforms:
            result, _ = transform(wav, sample_rate)
            corrupted.append(result)
        results[condition.name] = corrupted

    return results


def compute_bank_summary(per_condition_eer: dict[str, float]) -> CorruptionBankResult:
    """Compute summary statistics from per-condition EER values."""
    if not per_condition_eer:
        return CorruptionBankResult(clean_eer=0.0)

    eers = list(per_condition_eer.values())
    worst_name = max(per_condition_eer, key=per_condition_eer.get)

    # Group by family
    family_eers: dict[str, list[float]] = {}
    for name, eer in per_condition_eer.items():
        family = name.split("_")[0]
        family_eers.setdefault(family, []).append(eer)

    per_family = {k: float(np.mean(v)) for k, v in family_eers.items()}

    return CorruptionBankResult(
        clean_eer=per_condition_eer.get("clean", 0.0),
        per_condition=per_condition_eer,
        per_family=per_family,
        worst_condition=worst_name,
        worst_eer=max(eers),
        mean_eer=float(np.mean(eers)),
    )
