"""Reproducibility utilities: seeding, environment capture, provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python, NumPy, PyTorch CPU, and CUDA (if available).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def enable_deterministic_mode() -> None:
    """Enable PyTorch deterministic algorithms where possible.

    Note: This may reduce performance. Use for validation, not training speed.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def capture_environment() -> dict:
    """Capture current environment for reproducibility records."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy_version": np.__version__,
    }

    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "unknown"
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()

    return env


def get_git_info() -> dict:
    """Get current git commit and dirty status."""
    info = {"commit": "unknown", "dirty": True, "branch": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["commit"] = result.stdout.strip()[:12]

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        info["dirty"] = bool(result.stdout.strip())

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        info["branch"] = result.stdout.strip()
    except Exception:
        pass
    return info


def compute_file_hash(path: Path) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_provenance(output_dir: Path, config: dict | None = None) -> Path:
    """Save full provenance record alongside experiment outputs."""
    record = {
        "environment": capture_environment(),
        "git": get_git_info(),
        "config": config,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "provenance.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path
