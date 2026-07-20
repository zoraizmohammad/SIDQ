# Baseline Reproduction Guide

## Overview

The official ArA-DF 2026 baseline uses:
- **Frontend:** XLS-R 300M (wav2vec2-xls-r-300m)
- **Backend:** AASIST spectro-temporal graph attention
- **Official Track 2 dev-test EER:** 27.21%
- **Official Track 2 final-test EER:** 27.11%

## Prerequisites

1. Install SIDQ in development mode:
```bash
pip install -e ".[dev,full]"
```

2. Download the official baseline checkpoint:
```bash
# The baseline model is hosted at:
# https://huggingface.co/ArabicSpeech/ArA-DF-Baseline
git lfs install
git clone https://huggingface.co/ArabicSpeech/ArA-DF-Baseline checkpoints/baseline
```

3. Download the dataset:
```bash
# Dataset hosted at:
# https://huggingface.co/datasets/ArabicSpeech/ArA-DF-2026
python scripts/download_official_data.py --config track2_devtest
```

## Running Baseline Inference

```bash
sidq infer \
  --checkpoint checkpoints/baseline \
  --config configs/experiment/baseline.yaml \
  --output artifacts/predictions/baseline_track2_devtest.csv
```

## Expected Results

| Split | Expected EER | Source |
|-------|-------------|--------|
| Track 2 dev-test | ~27.21% | Official model card |
| Track 2 final-test | ~27.11% | Official model card |

**Note:** Local reproduction may differ slightly due to:
- Framework version differences
- Floating-point precision
- Hardware-specific behavior
- Batch processing differences

A locally measured EER within ±0.5% of the official value confirms correct
baseline operation.

## Verifying Score Direction

After generating predictions, verify score direction:

```bash
python -c "
from sidq.evaluation.metrics import validate_score_direction
import pandas as pd
import numpy as np

# Only possible on labeled splits (train/dev)
preds = pd.read_csv('artifacts/predictions/baseline_dev.csv')
labels = pd.read_csv('data/manifests/dev_labels.csv')
merged = preds.merge(labels, on='audio_id')
validate_score_direction(merged['logit'].values, merged['label'].values)
print('Score direction: CORRECT')
"
```

## Training from Scratch

To train a baseline-equivalent model:

```bash
sidq train configs/experiment/baseline.yaml
```

This uses:
- XLS-R 300M frozen frontend
- AASIST backend
- No augmentation (clean training)
- Cross-entropy loss
- AdamW optimizer with cosine schedule
- 30 epochs with early stopping (patience=5)

## Configuration

See `configs/experiment/baseline.yaml` for the full configuration.
The baseline config mirrors the official architecture as closely as possible.
