# ArA-DF 2026 — Starting Kit

## Submission format

Create a ZIP with the predictions CSV at the root:

```
submission.zip
└── track1_preds.csv    # (or track2_preds.csv for Track 2)
```

### CSV format

```csv
audio_id,logit
<utterance_id>,<float_score>
```

* `audio_id` must match **every** ID in the phase set exactly.
* `logit` is the bonafide-class score — higher = more likely bonafide.

## Phases

| Phase | Purpose |
|-------|---------|
| Development (10 % of test) | Public feedback on EER |
| Evaluation  (90 % of test) | Final ranking |

## Baseline

https://huggingface.co/ArabicSpeech/ArA-DF-Baseline

Use `run_inference.py` bundled with the model to generate predictions.

## Contact

yassine.el_kheir@dfki.de
