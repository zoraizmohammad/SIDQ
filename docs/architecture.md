# Architecture — SIDQ-T2

## System Overview

SIDQ-T2 is a three-branch ensemble for robust Arabic speech deepfake detection:

```
                        ┌─────────────┐
     Raw Audio ─────────┤  Branch A   ├───── Score A ──┐
     (16kHz mono)       │  XLS-R +    │                │
                        │  AASIST     │                │
                        └─────────────┘                │
                                                       │    ┌────────────┐
                        ┌─────────────┐                ├────┤   Fusion   ├─── Final Score
     Raw Audio ─────────┤  Branch B   ├───── Score B ──┤    │  (weighted │
                        │  WavLM +    │                │    │   mean)    │
                        │  Attentive  │                │    └────────────┘
                        └─────────────┘                │
                                                       │
                        ┌─────────────┐                │
     Raw Audio ─────────┤  Branch C   ├───── Score C ──┘
                        │  Raw Sinc   │
                        │  Specialist │
                        └─────────────┘
```

## Score Convention

- **Label 1** = bonafide (genuine human speech)
- **Label 0** = spoofed (synthetic/converted speech)
- **Higher score** = more likely bonafide
- **Primary metric**: Equal Error Rate (lower is better)

## Module Organization

| Module | Purpose |
|--------|---------|
| `sidq.data` | Dataset loading, manifests, validation splitting |
| `sidq.audio` | Audio I/O, resampling, cropping |
| `sidq.augment` | Corruption transforms and curriculum |
| `sidq.models` | SSL frontends, backends, raw specialist |
| `sidq.training` | Training engine with EER validation |
| `sidq.evaluation` | EER, metrics, corruption bank, analysis |
| `sidq.fusion` | Score normalization and weight optimization |
| `sidq.inference` | Scoring, multi-crop, ensemble, submission |
| `sidq.aws` | Optional Polly/Transcribe stress testing |
| `sidq.experiments` | Registry and experiment tracking |

## Key Design Decisions

1. **Score direction is enforced everywhere** — the bonafide-class logit (index 1)
   is always the submitted score.

2. **Corruption curriculum** — models are hardened through staged augmentation
   (clean → mild → robust → hard) to generalize across degradation conditions.

3. **Deterministic evaluation** — a fixed corruption bank with known seeds enables
   reproducible per-condition EER analysis.

4. **Fusion preserves direction** — normalization and weight optimization are
   constrained to maintain the bonafide-score ordering.
