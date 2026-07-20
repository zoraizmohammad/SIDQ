"""Regression tests for submission format — must never submit invalid files."""

import numpy as np
import pandas as pd
import pytest

from sidq.inference.submission import (
    SubmissionValidationError,
    build_submission,
    inspect_submission,
    validate_predictions,
)

REFERENCE_IDS = {f"test_{i:07d}" for i in range(100)}


class TestValidatePredictions:
    def test_valid_submission(self):
        df = pd.DataFrame({
            "audio_id": list(REFERENCE_IDS),
            "logit": np.random.randn(100).tolist(),
        })
        errors = validate_predictions(df, REFERENCE_IDS)
        assert errors == []

    def test_missing_ids(self):
        df = pd.DataFrame({
            "audio_id": list(REFERENCE_IDS)[:50],
            "logit": np.random.randn(50).tolist(),
        })
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("Missing" in e for e in errors)

    def test_extra_ids(self):
        ids = list(REFERENCE_IDS) + ["unknown_extra"]
        df = pd.DataFrame({
            "audio_id": ids,
            "logit": np.random.randn(101).tolist(),
        })
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("Unknown" in e for e in errors)

    def test_duplicate_ids(self):
        ids = list(REFERENCE_IDS)
        ids[0] = ids[1]  # Duplicate
        df = pd.DataFrame({
            "audio_id": ids,
            "logit": np.random.randn(100).tolist(),
        })
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("Duplicate" in e for e in errors)

    def test_nan_scores(self):
        scores = np.random.randn(100).tolist()
        scores[5] = float("nan")
        df = pd.DataFrame({"audio_id": list(REFERENCE_IDS), "logit": scores})
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("NaN" in e for e in errors)

    def test_inf_scores(self):
        scores = np.random.randn(100).tolist()
        scores[3] = float("inf")
        df = pd.DataFrame({"audio_id": list(REFERENCE_IDS), "logit": scores})
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("Inf" in e for e in errors)

    def test_hard_labels_rejected(self):
        df = pd.DataFrame({
            "audio_id": list(REFERENCE_IDS),
            "logit": [0.0 if i % 2 == 0 else 1.0 for i in range(100)],
        })
        errors = validate_predictions(df, REFERENCE_IDS)
        assert any("continuous" in e for e in errors)


class TestBuildSubmission:
    def test_creates_valid_zip(self, tmp_path):
        df = pd.DataFrame({
            "audio_id": list(REFERENCE_IDS),
            "logit": np.random.randn(100).tolist(),
        })
        out = tmp_path / "submission.zip"
        build_submission(df, REFERENCE_IDS, out)
        assert out.exists()

    def test_strict_raises_on_errors(self, tmp_path):
        df = pd.DataFrame({"audio_id": ["x"], "logit": [1.0]})
        with pytest.raises(SubmissionValidationError):
            build_submission(df, REFERENCE_IDS, tmp_path / "bad.zip", strict=True)


class TestInspectSubmission:
    def test_inspect_valid(self, tmp_path):
        df = pd.DataFrame({
            "audio_id": list(REFERENCE_IDS),
            "logit": np.random.randn(100).tolist(),
        })
        out = tmp_path / "submission.zip"
        build_submission(df, REFERENCE_IDS, out)
        info = inspect_submission(out)
        assert info["valid"] is True
        assert info["num_rows"] == 100

    def test_inspect_missing_file(self, tmp_path):
        info = inspect_submission(tmp_path / "nonexistent.zip")
        assert info["valid"] is False
