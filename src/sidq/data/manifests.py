"""Manifest generation and loading from Hugging Face metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sidq.data.schemas import ManifestEntry, SplitManifest


def load_hf_metadata(
    dataset_name: str = "ArabicSpeech/ArA-DF-2026",
    config: str = "track2_devtest",
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load dataset metadata from Hugging Face.

    Uses the datasets library to fetch parquet metadata.
    Falls back to local cache if available.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, config, cache_dir=str(cache_dir) if cache_dir else None)
        if hasattr(ds, "to_list"):
            return ds.to_list()
        split_name = list(ds.keys())[0]
        return [dict(row) for row in ds[split_name]]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HF metadata for {dataset_name}/{config}: {e}. "
            f"Ensure network access and valid dataset configuration."
        ) from e


def build_manifest_from_local(
    data_dir: Path,
    split_name: str,
    label_col: str = "label",
    metadata_file: str | None = None,
) -> SplitManifest:
    """Build a manifest from local FLAC files with optional metadata."""
    entries: list[ManifestEntry] = []

    if metadata_file:
        meta_path = data_dir / metadata_file
        if meta_path.suffix == ".json":
            with open(meta_path) as f:
                records = json.load(f)
        elif meta_path.suffix == ".jsonl":
            with open(meta_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported metadata format: {meta_path.suffix}")

        for rec in records:
            entries.append(
                ManifestEntry(
                    audio_id=rec["audio_id"],
                    relative_path=rec.get("path", f"{rec['audio_id']}.flac"),
                    label=rec.get(label_col),
                    duration_sec=rec.get("duration_sec"),
                    split=split_name,
                )
            )
    else:
        for flac_path in sorted(data_dir.glob("**/*.flac")):
            audio_id = flac_path.stem
            entries.append(
                ManifestEntry(
                    audio_id=audio_id,
                    relative_path=str(flac_path.relative_to(data_dir)),
                    split=split_name,
                )
            )

    return SplitManifest(
        split_name=split_name,
        entries=entries,
        checksum=compute_manifest_hash(entries),
    )


def compute_manifest_hash(entries: list[ManifestEntry]) -> str:
    """Compute deterministic hash of manifest entries."""
    content = json.dumps(
        [{"id": e.audio_id, "path": e.relative_path, "label": e.label} for e in entries],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_manifest(manifest: SplitManifest, path: Path) -> None:
    """Save manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)


def load_manifest(path: Path) -> SplitManifest:
    """Load manifest from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return SplitManifest(**data)
