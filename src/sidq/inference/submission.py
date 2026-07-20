"""Submission builder and validator for Codabench."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sidq.constants import SUBMISSION_COLUMNS, TRACK2_SUBMISSION_FILENAME


class SubmissionValidationError(Exception):
    """Raised when a submission fails validation."""

    pass


def validate_predictions(
    predictions: pd.DataFrame,
    reference_ids: set[str] | list[str],
) -> list[str]:
    """Validate predictions against requirements. Returns list of errors."""
    errors: list[str] = []
    reference_set = set(reference_ids)

    # Column check
    if list(predictions.columns) != SUBMISSION_COLUMNS:
        errors.append(
            f"Expected columns {SUBMISSION_COLUMNS}, got {list(predictions.columns)}"
        )
        return errors

    # Audio ID checks
    pred_ids = set(predictions["audio_id"])
    missing = reference_set - pred_ids
    if missing:
        errors.append(f"Missing {len(missing)} audio_ids. First few: {list(missing)[:5]}")

    extra = pred_ids - reference_set
    if extra:
        errors.append(f"Unknown {len(extra)} audio_ids. First few: {list(extra)[:5]}")

    duplicates = predictions[predictions["audio_id"].duplicated()]["audio_id"].tolist()
    if duplicates:
        errors.append(f"Duplicate audio_ids: {duplicates[:5]}")

    # Score checks
    logits = predictions["logit"]
    if logits.isna().any():
        n_nan = int(logits.isna().sum())
        errors.append(f"Found {n_nan} NaN values in logit column")

    if np.isinf(logits.values).any():
        errors.append("Found Inf values in logit column")

    unique_values = logits.nunique()
    if unique_values <= 2:
        errors.append(
            f"Only {unique_values} unique logit values. "
            "Submissions must be continuous scores, not hard labels."
        )

    # Row count
    if len(predictions) != len(reference_set):
        errors.append(
            f"Expected {len(reference_set)} rows, got {len(predictions)}"
        )

    return errors


def build_submission(
    predictions: pd.DataFrame,
    reference_ids: set[str] | list[str],
    output_path: Path,
    strict: bool = True,
) -> Path:
    """Build a valid submission ZIP.

    Args:
        predictions: DataFrame with audio_id and logit columns.
        reference_ids: Expected set of audio_ids.
        output_path: Path to write the submission.zip.
        strict: If True, raises on validation errors.

    Returns:
        Path to the created ZIP file.
    """
    errors = validate_predictions(predictions, reference_ids)
    if errors and strict:
        raise SubmissionValidationError(
            "Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_content = predictions[SUBMISSION_COLUMNS].to_csv(index=False)

    # Create ZIP
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(TRACK2_SUBMISSION_FILENAME, csv_content)

    return output_path


def inspect_submission(zip_path: Path) -> dict:
    """Inspect a submission ZIP and report contents."""
    info: dict = {"path": str(zip_path), "valid": False, "errors": []}

    if not zip_path.exists():
        info["errors"].append("File does not exist")
        return info

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            info["files"] = names

            if TRACK2_SUBMISSION_FILENAME not in names:
                info["errors"].append(
                    f"Missing {TRACK2_SUBMISSION_FILENAME} in archive"
                )
                return info

            if len(names) > 1:
                info["errors"].append(f"Extra files in archive: {names}")

            with zf.open(TRACK2_SUBMISSION_FILENAME) as f:
                df = pd.read_csv(f)
                info["num_rows"] = len(df)
                info["columns"] = list(df.columns)
                info["score_min"] = float(df["logit"].min())
                info["score_max"] = float(df["logit"].max())
                info["score_mean"] = float(df["logit"].mean())
                info["unique_scores"] = int(df["logit"].nunique())

    except zipfile.BadZipFile:
        info["errors"].append("Not a valid ZIP file")
        return info

    if not info["errors"]:
        info["valid"] = True

    return info
