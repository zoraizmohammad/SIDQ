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

## License

MIT
