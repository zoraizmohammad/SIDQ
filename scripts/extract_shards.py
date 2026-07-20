#!/usr/bin/env python3
"""Extract WebDataset TAR shards to get raw FLAC files.

Usage:
    python scripts/extract_shards.py --data-root ArA-DF-2026/data --splits track-2_test
    python scripts/extract_shards.py --data-root ArA-DF-2026/data --splits all
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ALL_SPLITS = [
    "train", "dev",
    "track-1_development_test", "track-1_test",
    "track-2_development_test", "track-2_test",
]


def extract_split(data_root: Path, split: str) -> int:
    """Extract all TAR files in a split directory. Returns file count."""
    split_dir = data_root / split
    if not split_dir.exists():
        print(f"  SKIP: {split_dir} does not exist")
        return 0

    tar_files = sorted(split_dir.glob("*.tar"))
    if not tar_files:
        print(f"  SKIP: No .tar files in {split_dir}")
        return 0

    # Check if already extracted
    flac_files = list(split_dir.rglob("*.flac"))
    if flac_files:
        print(f"  Already extracted: {len(flac_files)} FLAC files found")
        return len(flac_files)

    total_extracted = 0
    for tar_path in tar_files:
        try:
            with tarfile.open(tar_path, "r") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".flac")]
                tf.extractall(path=split_dir, members=members)
                total_extracted += len(members)
        except Exception as e:
            print(f"  ERROR extracting {tar_path.name}: {e}")

    print(f"  Extracted {total_extracted} files from {len(tar_files)} TARs")
    return total_extracted


def main():
    parser = argparse.ArgumentParser(description="Extract TAR shards")
    parser.add_argument("--data-root", type=Path, default=Path("ArA-DF-2026/data"))
    parser.add_argument("--splits", nargs="+", default=["all"],
                       help="Splits to extract (or 'all')")
    args = parser.parse_args()

    splits = ALL_SPLITS if "all" in args.splits else args.splits

    for split in splits:
        print(f"Processing: {split}")
        extract_split(args.data_root, split)


if __name__ == "__main__":
    main()
