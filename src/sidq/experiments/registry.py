"""Experiment registry for tracking all training runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentRecord:
    """Machine-readable record of a single experiment."""

    experiment_id: str
    git_commit: str = ""
    dirty_tree: bool = False
    timestamp: str = ""
    config_hash: str = ""
    manifest_hash: str = ""
    model_name: str = ""
    frontend: str = ""
    backend: str = ""
    seed: int = 42
    augmentation_profile: str = "none"
    epochs_trained: int = 0
    best_epoch: int = 0
    clean_eer: float | None = None
    corrupted_eer: float | None = None
    worst_group_eer: float | None = None
    checkpoint_path: str = ""
    prediction_path: str = ""
    notes: str = ""
    failed: bool = False
    hardware: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ExperimentRegistry:
    """File-based experiment registry using JSONL format."""

    def __init__(self, registry_path: Path | None = None):
        self.registry_path = registry_path or Path("artifacts/experiments/registry.jsonl")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, experiment: ExperimentRecord) -> None:
        """Append an experiment record to the registry."""
        if not experiment.timestamp:
            experiment.timestamp = datetime.now(timezone.utc).isoformat()
        if not experiment.git_commit:
            experiment.git_commit = self._get_git_commit()
            experiment.dirty_tree = self._is_dirty()

        with open(self.registry_path, "a") as f:
            f.write(experiment.to_json() + "\n")

    def load_all(self) -> list[ExperimentRecord]:
        """Load all experiment records."""
        if not self.registry_path.exists():
            return []
        records = []
        with open(self.registry_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    records.append(ExperimentRecord(**data))
        return records

    def get_best(self, metric: str = "corrupted_eer") -> ExperimentRecord | None:
        """Get the best experiment by a metric (lower is better for EER)."""
        records = [r for r in self.load_all() if not r.failed]
        if not records:
            return None
        valid = [r for r in records if getattr(r, metric) is not None]
        if not valid:
            return None
        return min(valid, key=lambda r: getattr(r, metric))

    def _get_git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def _is_dirty(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            return bool(result.stdout.strip())
        except Exception:
            return True


def compute_config_hash(config: dict) -> str:
    """Compute deterministic hash of experiment configuration."""
    content = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]
