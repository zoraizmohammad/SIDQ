"""Data schemas and manifest structures."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator

from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED


class AudioSample(BaseModel):
    """Single audio sample metadata."""

    audio_id: str
    path: Path | None = None
    label: int | None = None
    duration_sec: float | None = None
    sample_rate: int = 16_000
    split: str = ""
    dialect: str | None = None
    gender: str | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int | None) -> int | None:
        if v is not None and v not in (LABEL_BONAFIDE, LABEL_SPOOFED):
            raise ValueError(f"Label must be {LABEL_BONAFIDE} or {LABEL_SPOOFED}, got {v}")
        return v

    @field_validator("audio_id")
    @classmethod
    def validate_audio_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("audio_id cannot be empty")
        return v.strip()


class ManifestEntry(BaseModel):
    """Entry in a dataset manifest file."""

    audio_id: str
    relative_path: str
    label: int | None = None
    duration_sec: float | None = None
    split: str = ""


class SplitManifest(BaseModel):
    """Complete manifest for a dataset split."""

    split_name: str
    entries: list[ManifestEntry]
    checksum: str | None = None
    version: str = "1.0"

    @property
    def num_samples(self) -> int:
        return len(self.entries)

    @property
    def audio_ids(self) -> set[str]:
        return {e.audio_id for e in self.entries}

    def validate_no_duplicates(self) -> list[str]:
        """Return list of duplicate audio_ids, empty if none."""
        seen: dict[str, int] = {}
        for entry in self.entries:
            seen[entry.audio_id] = seen.get(entry.audio_id, 0) + 1
        return [k for k, v in seen.items() if v > 1]

    def validate_labels(self) -> list[str]:
        """Return IDs with missing labels in a labeled split."""
        return [e.audio_id for e in self.entries if e.label is None]
