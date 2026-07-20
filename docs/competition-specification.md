# Competition Specification — ArA-DF 2026 Track 2

## Task

Binary classification of Arabic speech audio:
- **Bonafide (label 1):** Genuine human speech from native Arabic speakers
- **Spoofed (label 0):** Synthetic speech from TTS or voice-conversion systems

## Track 2: Acoustic Robustness

Systems are evaluated on audio processed with real-world degradations:
codec compression, background noise, reverberation, re-recording effects,
and social-media re-encoding.

## Submission Format

```
submission.zip
└── track2_preds.csv
```

CSV columns:
- `audio_id`: must match every ID in the phase reference set exactly
- `logit`: bonafide-class score (continuous float)
  - Higher score = more likely bonafide
  - This is the bonafide-class logit from DeepFense CrossEntropy.get_score

## Scoring

- **Primary metric:** Equal Error Rate (EER) — lower is better
- **Secondary:** Accuracy, macro-F1 (reported, not ranked)
- Threshold-independent: EER does not require a fixed decision boundary

## Label Convention

| Label | Meaning     | Score direction              |
|-------|-------------|------------------------------|
| 1     | Bonafide    | Higher logit → more bonafide |
| 0     | Spoofed     | Lower logit → more spoofed   |

## Dataset Splits

| Split                     | Rows    | Labels available |
|---------------------------|---------|------------------|
| Train                     | 22,500  | Yes              |
| Dev                       | 21,000  | Yes              |
| Track 2 development-test  | 14,193  | No               |
| Track 2 final-test        | 127,746 | No               |

Audio: 16 kHz mono PCM, lossless FLAC, WebDataset TAR shards.
Metadata: Hugging Face Parquet configs.

## Baseline

- Model: XLS-R 300M frontend + AASIST backend
- Track 2 development-test EER: 27.21%
- Track 2 final-test EER: 27.11%
- Repository: https://huggingface.co/ArabicSpeech/ArA-DF-Baseline
- Dataset: https://huggingface.co/datasets/ArabicSpeech/ArA-DF-2026

## Submission Phases

### Development Phase (June 16 – July 20, 2026)
- Reference set: 14,193 utterances (10% of test)
- Up to 100 submissions total, 10 per day
- Public feedback leaderboard

### Evaluation Phase (July 20 – July 25, 2026)
- Reference set: 127,746 utterances (90% of test)
- Up to 3 system submissions (best EER is official)
- Final ranking

## Rules and Constraints

- Do NOT submit hard labels (only continuous scores)
- Do NOT use official test set for supervised training
- Do NOT infer labels from filenames or ordering
- Do NOT leak leaderboard feedback into validation selection
- System description papers required for final ranking eligibility

## Key Dates

- July 20, 2026: Development phase ends
- July 20–25, 2026: Evaluation phase (final-test submissions)
- July 25, 2026: Leaderboard freeze
- August 8, 2026: System papers due

## Sources

- Official website: ArA-DF 2026 (ArabicNLP 2026 Shared Task 14)
- Codabench Track 2 page
- Hugging Face dataset card
- Hugging Face baseline model card
- Starting kit README (included in repository)
