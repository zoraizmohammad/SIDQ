# SIDQ-T2

**Spoof Identification under Degraded Quality — Track 2**

Corruption-Aware Self-Supervised Fusion for Robust Arabic Speech Deepfake Detection.

## Overview

SIDQ-T2 is a competition system for [ArA-DF 2026 Track 2: Acoustic Robustness](https://arabicnlp.github.io/ArA-DF-2026/),
which evaluates Arabic speech deepfake detectors under practical audio degradations
including codec compression, background noise, and re-recording effects.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
make test

# Check code quality
make lint
make type-check
```

## Project Structure

```
src/sidq/          Core library
  data/            Data loading and manifests
  audio/           Audio I/O and preprocessing
  augment/         Corruption and augmentation transforms
  models/          Model architectures (XLS-R, WavLM, Raw)
  training/        Training engine and optimization
  evaluation/      EER metrics and analysis
  fusion/          Score normalization and model fusion
  inference/       Batch inference and submission
  aws/             Optional Polly/Transcribe integration
  experiments/     Experiment registry and tracking
configs/           Hydra/OmegaConf YAML configurations
scripts/           Standalone execution scripts
tests/             Unit, integration, and regression tests
docs/              Documentation and paper outline
```

## Task

Classify Arabic speech audio as:
- **Bonafide (1):** genuine human speech
- **Spoofed (0):** synthetic TTS or voice-converted speech

Submit continuous bonafide-class scores. Higher score = more likely bonafide.
Ranked by Equal Error Rate (lower is better).

## Training

```bash
# Smoke test (CPU, no data required)
sidq train configs/experiment/smoke.yaml

# Baseline reproduction
sidq train configs/experiment/baseline.yaml

# Evaluate a checkpoint
sidq evaluate checkpoints/best.pt --config configs/experiment/baseline.yaml
```

## Inference and Submission

```bash
# Run inference
sidq infer checkpoints/best.pt --output artifacts/predictions/track2_preds.csv

# Build submission ZIP
python -c "
from sidq.inference.submission import build_submission
import pandas as pd
preds = pd.read_csv('artifacts/predictions/track2_preds.csv')
ref_ids = set(preds['audio_id'])
build_submission(preds, ref_ids, Path('artifacts/submissions/submission.zip'))
"
```

## AWS Stress Testing (Optional)

```bash
sidq aws stress-test \
  --polly-config configs/aws/polly.yaml \
  --budget-usd 5 \
  --dry-run
```

## Documentation

- [Architecture](docs/architecture.md)
- [Competition Specification](docs/competition-specification.md)
- [Baseline Reproduction](docs/baseline-reproduction.md)
- [System Paper Outline](docs/system-paper-outline.md)

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT
