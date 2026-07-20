"""Deterministic validation splitting with stratification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from sidq.data.schemas import ManifestEntry, SplitManifest


@dataclass
class SplitConfig:
    """Configuration for train/validation splitting."""

    val_fraction: float = 0.15
    seed: int = 42
    stratify_by_label: bool = True
    group_by_prefix: bool = False
    prefix_separator: str = "_"
    version: str = "1.0"


def create_validation_split(
    manifest: SplitManifest,
    config: SplitConfig | None = None,
) -> tuple[SplitManifest, SplitManifest]:
    """Split a labeled manifest into train and validation sets.

    Returns:
        Tuple of (train_manifest, val_manifest)
    """
    if config is None:
        config = SplitConfig()
    entries = manifest.entries
    labeled = [e for e in entries if e.label is not None]

    if not labeled:
        raise ValueError("Cannot split: no labeled entries found")

    rng = np.random.default_rng(config.seed)

    if config.stratify_by_label:
        train_entries, val_entries = _stratified_split(labeled, config.val_fraction, rng)
    else:
        indices = rng.permutation(len(labeled))
        n_val = int(len(labeled) * config.val_fraction)
        val_indices = set(indices[:n_val].tolist())
        train_entries = [labeled[i] for i in range(len(labeled)) if i not in val_indices]
        val_entries = [labeled[i] for i in val_indices]

    _verify_no_overlap(train_entries, val_entries)

    train_manifest = SplitManifest(
        split_name=f"{manifest.split_name}_train",
        entries=train_entries,
        version=config.version,
    )
    val_manifest = SplitManifest(
        split_name=f"{manifest.split_name}_val",
        entries=val_entries,
        version=config.version,
    )

    return train_manifest, val_manifest


def _stratified_split(
    entries: list[ManifestEntry],
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Split maintaining label proportions."""
    by_label: dict[int, list[ManifestEntry]] = {}
    for e in entries:
        label = e.label
        assert label is not None
        by_label.setdefault(label, []).append(e)

    train_all: list[ManifestEntry] = []
    val_all: list[ManifestEntry] = []

    for label in sorted(by_label.keys()):
        group = by_label[label]
        indices = rng.permutation(len(group))
        n_val = max(1, int(len(group) * val_fraction))
        val_idx = set(indices[:n_val].tolist())

        for i, entry in enumerate(group):
            if i in val_idx:
                val_all.append(entry)
            else:
                train_all.append(entry)

    return train_all, val_all


def _verify_no_overlap(
    train: list[ManifestEntry], val: list[ManifestEntry]
) -> None:
    """Assert no ID overlap between splits."""
    train_ids = {e.audio_id for e in train}
    val_ids = {e.audio_id for e in val}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(
            f"Data leakage: {len(overlap)} IDs appear in both splits. "
            f"First few: {list(overlap)[:5]}"
        )


def compute_split_hash(train: SplitManifest, val: SplitManifest) -> str:
    """Compute deterministic hash of a split for versioning."""
    data = {
        "train_ids": sorted(e.audio_id for e in train.entries),
        "val_ids": sorted(e.audio_id for e in val.entries),
    }
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
