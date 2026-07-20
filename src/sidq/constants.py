"""Project-wide constants for SIDQ-T2."""

# Label convention: matches ArA-DF 2026 official specification
LABEL_BONAFIDE: int = 1
LABEL_SPOOFED: int = 0

# Score direction: higher logit = more likely bonafide
# This is critical for EER computation and submission correctness
SCORE_DIRECTION = "higher_is_bonafide"

# Audio parameters
SAMPLE_RATE: int = 16_000
MONO_CHANNELS: int = 1

# Dataset split names
SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TRACK2_DEVTEST = "track2_devtest"
SPLIT_TRACK2_FINALTEST = "track2_finaltest"

# Expected row counts from official specification
EXPECTED_ROWS = {
    SPLIT_TRAIN: 22_500,
    SPLIT_DEV: 21_000,
    SPLIT_TRACK2_DEVTEST: 14_193,
    SPLIT_TRACK2_FINALTEST: 127_746,
}

# Submission file naming
TRACK2_SUBMISSION_FILENAME = "track2_preds.csv"
SUBMISSION_COLUMNS = ["audio_id", "logit"]
