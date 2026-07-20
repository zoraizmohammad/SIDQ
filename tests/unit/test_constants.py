"""Tests for project constants and label conventions."""

from sidq.constants import (
    EXPECTED_ROWS,
    LABEL_BONAFIDE,
    LABEL_SPOOFED,
    SAMPLE_RATE,
    SCORE_DIRECTION,
    SUBMISSION_COLUMNS,
    TRACK2_SUBMISSION_FILENAME,
)


def test_label_values():
    assert LABEL_BONAFIDE == 1
    assert LABEL_SPOOFED == 0
    assert LABEL_BONAFIDE != LABEL_SPOOFED


def test_score_direction():
    assert SCORE_DIRECTION == "higher_is_bonafide"


def test_sample_rate():
    assert SAMPLE_RATE == 16_000


def test_expected_rows():
    assert EXPECTED_ROWS["train"] == 22_500
    assert EXPECTED_ROWS["dev"] == 21_000
    assert EXPECTED_ROWS["track2_devtest"] == 14_193
    assert EXPECTED_ROWS["track2_finaltest"] == 127_746


def test_submission_format():
    assert TRACK2_SUBMISSION_FILENAME == "track2_preds.csv"
    assert SUBMISSION_COLUMNS == ["audio_id", "logit"]
