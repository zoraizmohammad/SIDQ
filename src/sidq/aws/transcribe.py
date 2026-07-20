"""Amazon Transcribe integration for transcript-stability analysis. Optional."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sidq.aws.config import AWSConfig


@dataclass
class TranscribeResult:
    """Result from a transcription job."""

    text: str
    confidence: float
    job_name: str
    cost_estimate_usd: float = 0.0


def estimate_transcribe_cost(duration_sec: float) -> float:
    """Estimate transcription cost.

    Standard: $0.024 per minute (first 250k minutes/month).
    """
    minutes = duration_sec / 60.0
    return minutes * 0.024


def transcribe_audio(
    audio_path: Path,
    config: AWSConfig,
    job_name: str | None = None,
    language_code: str = "ar-SA",
) -> TranscribeResult:
    """Transcribe audio via Amazon Transcribe.

    For diagnostic analysis only — not for label inference.
    """
    config.validate_enabled()

    if config.dry_run:
        return TranscribeResult(
            text="[dry-run: no transcription performed]",
            confidence=0.0,
            job_name=job_name or "dry-run",
            cost_estimate_usd=0.0,
        )

    raise NotImplementedError(
        "Live Transcribe integration requires S3 upload and job polling. "
        "Use dry_run=True for testing, or implement the full pipeline."
    )


def normalize_arabic_text(text: str, normalize_alef: bool = True) -> str:
    """Normalize Arabic text for comparison.

    Handles diacritics, alef variants, and whitespace.
    """
    import re

    text = re.sub(r'[ً-ٰٟ]', '', text)

    if normalize_alef:
        text = re.sub(r'[آأإ]', 'ا', text)

    text = re.sub(r'[ـ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
