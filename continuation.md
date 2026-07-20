# Continuation Guide — SIDQ-T2

## Where We Left Off

The full SIDQ-T2 codebase is implemented (48 commits) and verified (176 tests passing).
Training was started but interrupted due to disk space constraints on the local machine.

## Current State

### Repository
- **Branch:** `main`
- **Latest commit:** `2f172c6` — "fix: audio loading fallback and MPS attention compat"
- **Total commits:** 48
- **Tests:** 176 passing, lint clean
- **Author:** Mohammad Zoraiz <zoraizmohammad@gmail.com>

### Data Downloaded (on local disk at `ArA-DF-2026/`)
- ✅ All metadata parquets: `ArA-DF-2026/metadata/*.parquet`
- ✅ Train audio extracted: 22,500 FLACs in `ArA-DF-2026/data/train/`
- ✅ Dev audio extracted: 21,000 FLACs in `ArA-DF-2026/data/dev/`
- ✅ Track 2 dev-test extracted: 14,193 FLACs in `ArA-DF-2026/data/track-2_development_test/`
- ✅ Track 2 final-test extracted: 127,746 FLACs in `ArA-DF-2026/data/track-2_test/`
- ✅ Baseline checkpoint: `ArA-DF-2026/models/ArA-DF-Baseline/best_model.pth` (3.5GB)
- ❌ Fairseq XLS-R weights: `ArA-DF-2026/models/ArA-DF-Baseline/xlsr2_300m.pt` NOT downloaded (download kept failing)
- ✅ Parquets built: `ArA-DF-2026/parquets/` (aradf_train, aradf_val, aradf_track-2_development_test)

### Submission Ready
- ✅ `artifacts/submissions/submission_baseline_track2.zip` — baseline predictions for 127,746 Track 2 final-test samples
- Expected EER: ~27.11% (official baseline)
- Upload to: https://www.codabench.org/competitions/17139/

### Training Status
- Smoke test passed: XLS-R + AASIST trained successfully (11.36% EER on tiny set)
- Full training was launched but interrupted (disk full + process killed)
- The `checkpoints/best_sidq.pt` file is from the smoke test only (not a real trained model)

---

## What To Do Next (on cloud desktop)

### Step 1: Set Up Environment
```bash
cd /path/to/SIDQ
pip install -e ".[dev,full]"
pip install deepfense  # For running official baseline inference
```

### Step 2: Download Data (if not copying from local)
```bash
python -c "
from huggingface_hub import snapshot_download
# Full dataset (~30GB)
snapshot_download('ArabicSpeech/ArA-DF-2026', repo_type='dataset', local_dir='ArA-DF-2026')
# Baseline model (~5GB)
snapshot_download('ArabicSpeech/ArA-DF-Baseline', local_dir='ArA-DF-2026/models/ArA-DF-Baseline')
"
```

### Step 3: Extract TAR Shards
```bash
python scripts/extract_shards.py --data-root ArA-DF-2026/data --splits all
```

### Step 4: Build Parquets (for DeepFense)
```bash
python ArA-DF-2026/models/ArA-DF-Baseline/build_parquets.py \
  --data_root ArA-DF-2026/data \
  --meta_root ArA-DF-2026/metadata \
  --output_dir ArA-DF-2026/parquets
```

### Step 5: Train with SIDQ (Our Code)
```bash
# With CUDA GPU (recommended)
python scripts/train_sidq.py \
  --data-root ArA-DF-2026/data \
  --meta-root ArA-DF-2026/metadata \
  --augmentation robust \
  --epochs 30 \
  --batch-size 16 \
  --patience 7 \
  --output-dir checkpoints

# For MPS (Apple Silicon) — add attn_implementation="eager" is already in code
# Same command works, just uses MPS automatically
```

**Note:** The training script (`scripts/train_sidq.py`) uses:
- HuggingFace transformers XLS-R 300M (`facebook/wav2vec2-xls-r-300m`)
- Our AASIST backend (`src/sidq/models/backends/aasist.py`)
- Frozen XLS-R frontend, only AASIST is trained (374K params)
- Class-weighted CE loss (0.33 spoofed, 0.67 bonafide for 2:1 imbalance)
- Cosine annealing LR schedule
- Early stopping on validation EER

### Step 6: Train with DeepFense (Official Framework)
```bash
# If DeepFense is installed and xlsr2_300m.pt is downloaded:
python scripts/train_baseline.py --data-root ArA-DF-2026 --build-parquets
```

### Step 7: Run Inference on Track 2 Final-Test
```bash
# With our trained model:
python scripts/run_track2_inference.py \
  --audio-dir ArA-DF-2026/data/track-2_test \
  --config ArA-DF-2026/models/ArA-DF-Baseline/config.yaml \
  --checkpoint checkpoints/best_sidq.pt \
  --output track2_preds_improved.csv

# With multi-crop for better robustness:
python scripts/run_track2_inference.py \
  --audio-dir ArA-DF-2026/data/track-2_test \
  --config ArA-DF-2026/models/ArA-DF-Baseline/config.yaml \
  --checkpoint checkpoints/best_sidq.pt \
  --output track2_preds_multicrop.csv \
  --multi-crop
```

### Step 8: Build Final Submission
```python
from pathlib import Path
import pandas as pd
from sidq.inference.submission import build_submission

preds = pd.read_csv('track2_preds_improved.csv')
ref_ids = set(pd.read_parquet('ArA-DF-2026/metadata/track-2_test.parquet')['id'])
build_submission(preds, ref_ids, Path('artifacts/submissions/submission_improved.zip'))
```

### Step 9: Submit to CodaBench
Upload to: https://www.codabench.org/competitions/17139/
- You have up to 3 submissions for the evaluation phase
- Best EER across all submissions is your official score
- Deadline: July 25, 2026

---

## Competition Key Facts

| Item | Value |
|------|-------|
| Task | Binary: bonafide (1) vs spoofed (0) |
| Score direction | Higher logit = more bonafide |
| Metric | EER (lower is better) |
| Track 2 focus | Acoustic robustness (codec, noise, re-recording) |
| Final-test size | 127,746 utterances |
| Submission format | ZIP containing `track2_preds.csv` with columns: `audio_id,logit` |
| Baseline EER | 27.11% on final-test |
| Codabench URL | https://www.codabench.org/competitions/17139/ |
| Evaluation phase | July 20–25, 2026 |
| Max submissions | 3 total |

---

## Training Strategy Priority

1. **First submission (ready now):** Upload `artifacts/submissions/submission_baseline_track2.zip`
2. **Second submission:** Train with `--augmentation robust`, run inference, submit
3. **Third submission:** Train with `--augmentation hard` or try multi-crop inference on the robust model

## Architecture

```
XLS-R 300M (frozen, 300M params) → AASIST Backend (trainable, 374K params) → 2-class logits
                                                                                    ↓
                                                                          logits[:, 1] = bonafide score
```

## File Locations

| Purpose | Path |
|---------|------|
| Training script | `scripts/train_sidq.py` |
| Inference script | `scripts/run_track2_inference.py` |
| Submission builder | `src/sidq/inference/submission.py` |
| AASIST backend | `src/sidq/models/backends/aasist.py` |
| Augmentation | `src/sidq/augment/curriculum.py` |
| EER metric | `src/sidq/evaluation/eer.py` |
| Existing baseline preds | `track2_preds.csv` (127,746 rows) |
| Ready submission | `artifacts/submissions/submission_baseline_track2.zip` |
