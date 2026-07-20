#!/usr/bin/env python3
"""Train the baseline XLS-R + AASIST model on ArA-DF 2026.

This script uses DeepFense framework if available, otherwise falls back
to the SIDQ native training pipeline.

Usage:
    python scripts/train_baseline.py --data-root ArA-DF-2026

Requirements:
    - DeepFense installed (pip install deepfense)
    - ArA-DF-2026 dataset downloaded and extracted
    - ArA-DF-Baseline model downloaded (for XLS-R weights)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def setup_deepfense_config(data_root: Path, model_root: Path) -> Path:
    """Create a configured DeepFense config.yaml for training."""
    from omegaconf import OmegaConf

    config_template = model_root / "config.yaml"
    if not config_template.exists():
        raise FileNotFoundError(f"Config template not found: {config_template}")

    cfg = OmegaConf.load(config_template)

    # Set paths
    parquets_dir = data_root / "parquets"
    xlsr_path = model_root / "xlsr2_300m.pt"

    cfg.model.frontend.args.ckpt_path = str(xlsr_path.resolve())
    cfg.data.train.parquet_files = [str(parquets_dir / "aradf_train.parquet")]
    cfg.data.val.parquet_files = [str(parquets_dir / "aradf_val.parquet")]
    cfg.data.test.parquet_files = [
        str(parquets_dir / "aradf_track-2_development_test.parquet"),
        str(parquets_dir / "aradf_track-2_test.parquet"),
    ]
    cfg.data.test.dataset_names = ["track-2_development_test", "track-2_test"]

    # Training config tuned for MPS/single-GPU
    cfg.training.epochs = 50
    cfg.training.device = "mps"  # Apple Silicon
    cfg.training.early_stopping_patience = 7
    cfg.training.optimizer.lr = 1e-6
    cfg.data.train.batch_size = 16  # Smaller for MPS memory
    cfg.data.val.batch_size = 32

    output_config = data_root / "configs" / "train_config.yaml"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_config)
    print(f"Config written to: {output_config}")
    return output_config


def build_parquets(data_root: Path, model_root: Path) -> None:
    """Build parquet files using the baseline's build_parquets.py."""
    import subprocess

    script = model_root / "build_parquets.py"
    if not script.exists():
        raise FileNotFoundError(f"build_parquets.py not found: {script}")

    parquets_dir = data_root / "parquets"
    parquets_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--data_root", str(data_root / "data"),
        "--meta_root", str(data_root / "metadata"),
        "--output_dir", str(parquets_dir),
    ]
    print(f"Building parquets: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("build_parquets.py failed")
    print(f"Parquets built in: {parquets_dir}")


def train_with_deepfense(config_path: Path) -> None:
    """Train using DeepFense framework."""
    import subprocess

    cmd = [sys.executable, "-m", "deepfense.train", "--config", str(config_path)]
    print(f"Starting training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--data-root", type=Path, default=Path("ArA-DF-2026"))
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--build-parquets", action="store_true", help="Build parquet files first")
    parser.add_argument("--config-only", action="store_true", help="Only generate config, don't train")
    args = parser.parse_args()

    if args.model_root is None:
        args.model_root = args.data_root / "models" / "ArA-DF-Baseline"

    if not args.model_root.exists():
        print(f"ERROR: Baseline model not found at {args.model_root}")
        print("Download with: python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('ArabicSpeech/ArA-DF-Baseline', local_dir='ArA-DF-2026/models/ArA-DF-Baseline')\"")
        sys.exit(1)

    if args.build_parquets:
        build_parquets(args.data_root, args.model_root)

    config_path = setup_deepfense_config(args.data_root, args.model_root)

    if args.config_only:
        print("Config generated. Run training manually with:")
        print(f"  python -m deepfense.train --config {config_path}")
        return

    train_with_deepfense(config_path)


if __name__ == "__main__":
    main()
