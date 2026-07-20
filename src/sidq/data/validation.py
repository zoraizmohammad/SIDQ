"""Dataset audit and validation utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sidq.data.schemas import SplitManifest


@dataclass
class DatasetAuditReport:
    """Summary statistics from a dataset audit."""

    split_name: str
    num_samples: int = 0
    label_counts: dict[int, int] = field(default_factory=dict)
    duration_stats: dict[str, float] = field(default_factory=dict)
    dialect_counts: dict[str, int] = field(default_factory=dict)
    gender_counts: dict[str, int] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    decode_failures: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.duplicate_ids) == 0 and len(self.missing_labels) == 0


def audit_manifest(manifest: SplitManifest) -> DatasetAuditReport:
    """Audit a manifest for data quality issues."""
    report = DatasetAuditReport(split_name=manifest.split_name)
    report.num_samples = manifest.num_samples
    report.duplicate_ids = manifest.validate_no_duplicates()

    label_counter: Counter[int] = Counter()
    durations: list[float] = []

    for entry in manifest.entries:
        if entry.label is not None:
            label_counter[entry.label] += 1
        else:
            report.missing_labels.append(entry.audio_id)

        if entry.duration_sec is not None:
            durations.append(entry.duration_sec)

    report.label_counts = dict(label_counter)

    if durations:
        arr = np.array(durations)
        report.duration_stats = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "total_hours": float(arr.sum() / 3600),
        }

    return report


def check_audio_file(path: Path) -> tuple[bool, str]:
    """Check if an audio file can be decoded successfully."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path))
        if data.size == 0:
            return False, "Empty audio file"
        if not np.isfinite(data).all():
            return False, "Non-finite values in audio"
        return True, ""
    except Exception as e:
        return False, str(e)


def compute_content_hash(path: Path) -> str:
    """Compute SHA-256 hash of file content for deduplication."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
