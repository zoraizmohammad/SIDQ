#!/usr/bin/env python3
"""Generate SIDQ_Colab.ipynb — a credit-optimized Colab training notebook.

Run once to (re)produce notebooks/SIDQ_Colab.ipynb. Keeping the notebook
generated from source makes it easy to review in a diff and regenerate.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/zoraizmohammad/SIDQ"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells: list[dict] = []

cells.append(md(
    "# SIDQ-T2 — ArA-DF 2026 Track 2 (Colab, credit-optimized)",
    "",
    "Trains **XLS-R 300M (frozen) + AASIST head (374K trainable params)** for Arabic",
    "speech deepfake detection, and builds a CodaBench submission ZIP.",
    "",
    "### Designed to stay well under your 100 compute units",
    "- **Use a T4 GPU** (Runtime ▸ Change runtime type ▸ T4). *Do not* pick A100 — it burns",
    "  units ~6× faster for **zero** benefit here (only a tiny head trains).",
    "- Frozen frontend ⇒ a full 30-epoch run is **~1–2 h on T4 ≈ a few compute units**.",
    "- Per-epoch validation uses a fixed **3,000-clip dev subset** (full dev only at the end).",
    "- A **wall-clock budget guard** (`--max-hours`) stops training before it can overrun.",
    "- Checkpoints go to **Google Drive** so a disconnect never wastes spent units.",
    "- Colab Pro does **not** auto-charge overage — when units hit 0 you drop to the free",
    "  tier. This notebook is built so you finish long before that.",
    "",
    "**Run cells top to bottom.** Each section says what it costs. Deadline: **Jul 25, 2026**.",
))

# ---- Section 1: GPU + credit estimator ----
cells.append(md(
    "## 1 · Check GPU & estimate credit burn",
    "Confirms a T4 is attached and prints an approximate units/hour rate so you can track",
    "spend. (Rates are Google's published approximations and can change.)",
))
cells.append(code(
    "import subprocess, torch",
    "print('CUDA available:', torch.cuda.is_available())",
    "assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU'",
    "name = torch.cuda.get_device_name(0)",
    "print('GPU:', name)",
    "print(subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used',",
    "                      '--format=csv'], capture_output=True, text=True).stdout)",
    "",
    "# Approx Colab compute-unit burn (units/hour). Google adjusts these over time.",
    "RATE = {'T4': 1.8, 'L4': 4.5, 'A100': 11.8, 'V100': 5.0}",
    "tier = next((k for k in RATE if k in name), None)",
    "if tier:",
    "    r = RATE[tier]",
    "    print(f'\\nApprox burn: ~{r} units/hr on {tier}.')",
    "    print(f'Your ~100 units => ~{100/r:.0f} GPU-hours of headroom.')",
    "    if tier in ('A100','L4'):",
    "        print('WARNING: this tier is expensive. Switch to T4 to conserve units.')",
    "else:",
    "    print('Unknown GPU tier; monitor units manually in the Colab resources panel.')",
))

# ---- Section 2: Drive ----
cells.append(md(
    "## 2 · Mount Google Drive (checkpoint persistence)",
    "Checkpoints and submissions are written under `MyDrive/sidq/` so a session",
    "disconnect never loses spent-unit progress. You can skip this, but then a",
    "disconnect wastes the units already burned.",
))
cells.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "import os",
    "DRIVE = '/content/drive/MyDrive/sidq'",
    "os.makedirs(DRIVE, exist_ok=True)",
    "print('Artifacts ->', DRIVE)",
))

# ---- Section 3: Clone + install ----
cells.append(md(
    "## 3 · Clone repo & install dependencies",
    "Colab runs modern Ubuntu (glibc ≥2.35), so everything installs from prebuilt wheels",
    "(unlike the old Amazon Linux 2 box where these had to build from source).",
    "`ffmpeg` is needed for codec augmentation. This cell uses **no GPU** — near-zero units.",
))
cells.append(code(
    "%cd /content",
    f"![ -d SIDQ ] || git clone {REPO_URL} SIDQ",
    "%cd /content/SIDQ",
    "!git pull --ff-only",
    "!apt-get -qq install -y ffmpeg >/dev/null",
    "# Core deps: torch/torchaudio are preinstalled on Colab. Add the rest.",
    "# (Don't hide output — we want to see any real pip error.)",
    "!pip install -e '.[full]' huggingface_hub hf_transfer",
    "",
    "# `pip install -e` writes a .pth that Python only reads at interpreter startup,",
    "# so the ALREADY-RUNNING kernel won't see `sidq` yet. Add src/ to sys.path now",
    "# to make the editable package importable without restarting the runtime.",
    "import sys",
    "if '/content/SIDQ/src' not in sys.path:",
    "    sys.path.insert(0, '/content/SIDQ/src')",
    "",
    "import sidq, torch, transformers",
    "print('sidq OK | torch', torch.__version__, '| transformers', transformers.__version__)",
))

# ---- Section 4: config ----
cells.append(md(
    "## 4 · Configuration",
    "Set what you want to run. Defaults are tuned for **cheapest useful** first pass.",
))
cells.append(code(
    "# --- What to run ---",
    "DO_BASELINE_INSURANCE = False  # DeepFense baseline on test set (optional floor; see §5)",
    "DO_TRAIN              = True   # Train SIDQ-native model (primary path)",
    "DO_INFER_SUBMIT       = True   # Score test set + build submission ZIP",
    "",
    "# --- Training knobs (credit-aware) ---",
    "AUGMENTATION = 'robust'   # 'none'|'mild'|'robust'|'hard'  (robust suits Track 2)",
    "EPOCHS       = 30         # early stopping usually ends sooner",
    "BATCH_SIZE   = 32         # T4 handles 32 for a frozen frontend",
    "PATIENCE     = 7",
    "VAL_SUBSET   = 3000       # per-epoch dev subset (full dev evaluated at end)",
    "MAX_HOURS    = 2.5        # HARD wall-clock guard -> caps units spent",
    "LR           = 1e-4",
    "",
    "# --- Inference knobs ---",
    "MULTI_CROP   = True       # average score over overlapping windows (more robust)",
    "",
    "# --- Which test split to score for submission ---",
    "TEST_SPLIT   = 'track-2_test'  # final eval phase (127,746 clips)",
    "",
    "# --- Shared helpers (Python API, so they never depend on `hf` being on PATH) ---",
    "import glob, tarfile",
    "from huggingface_hub import snapshot_download",
    "",
    "def fetch(patterns, repo='ArabicSpeech/ArA-DF-2026', repo_type='dataset',",
    "          local_dir='ArA-DF-2026'):",
    "    \"\"\"Download only the files matching `patterns` (list of glob patterns).\"\"\"",
    "    snapshot_download(repo, repo_type=repo_type, local_dir=local_dir,",
    "                      allow_patterns=patterns)",
    "",
    "def extract_split(split):",
    "    \"\"\"Extract FLACs from a split's TAR shards if not already extracted.\"\"\"",
    "    d = f'ArA-DF-2026/data/{split}'",
    "    if not glob.glob(f'{d}/**/*.flac', recursive=True):",
    "        for t in sorted(glob.glob(f'{d}/*.tar')):",
    "            with tarfile.open(t) as tf:",
    "                tf.extractall(d, [m for m in tf.getmembers() if m.name.endswith('.flac')])",
    "    n = len(glob.glob(f'{d}/**/*.flac', recursive=True))",
    "    print(f'{split}: {n} flac')",
    "    return n",
    "",
    "print('Config set.')",
))

# ---- Section 5: baseline insurance (optional) ----
cells.append(md(
    "## 5 · (Optional) DeepFense baseline — insurance submission",
    "Reproduces the official **~27.11% EER** baseline as a safety floor. This path needs",
    "`deepfense` + `fairseq` (to load `xlsr2_300m.pt`). On Colab these usually install",
    "cleanly; if `fairseq` fights the toolchain, **skip this** — the SIDQ-native path (§6–7)",
    "is the real improvement path and doesn't need either package.",
    "",
    "Downloads the baseline model (~7.6 GB) + the test split shards. Inference on 127k clips",
    "is ~20–40 min on T4 (**~1 unit**).",
))
cells.append(code(
    "if DO_BASELINE_INSURANCE:",
    "    import os",
    "    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'",
    "    !pip install -q deepfense fairseq",
    "    # Baseline model (config.yaml, best_model.pth, xlsr2_300m.pt, run_inference.py)",
    "    snapshot_download('ArabicSpeech/ArA-DF-Baseline',",
    "                      local_dir='ArA-DF-2026/models/ArA-DF-Baseline')",
    "else:",
    "    print('Skipping baseline insurance (DO_BASELINE_INSURANCE=False).')",
))
cells.append(code(
    "if DO_BASELINE_INSURANCE:",
    "    # Download + extract only the chosen test split's shards + metadata",
    "    fetch([f'data/{TEST_SPLIT}/*', f'metadata/{TEST_SPLIT}.parquet'])",
    "    extract_split(TEST_SPLIT)",
))
cells.append(code(
    "if DO_BASELINE_INSURANCE:",
    "    import subprocess, os",
    "    MODEL = os.path.abspath('ArA-DF-2026/models/ArA-DF-Baseline')",
    "    subprocess.run(['python', 'run_inference.py',",
    "                    '--audio_dir', f'../../data/{TEST_SPLIT}',",
    "                    '--config', f'{MODEL}/config.yaml',",
    "                    '--checkpoint', f'{MODEL}/best_model.pth',",
    "                    '--output', '/content/SIDQ/baseline_preds.csv'],",
    "                   cwd=MODEL, check=True)",
    "    import pandas as pd; from pathlib import Path",
    "    from sidq.inference.submission import build_submission, inspect_submission",
    "    preds = pd.read_csv('baseline_preds.csv')",
    "    ref = set(pd.read_parquet(f'ArA-DF-2026/metadata/{TEST_SPLIT}.parquet')['id'])",
    "    out = build_submission(preds, ref, Path(f'{DRIVE}/submission_baseline_track2.zip'))",
    "    print(inspect_submission(out))",
))

# ---- Section 6: download train/dev + train ----
cells.append(md(
    "## 6 · Download train/dev & train the SIDQ model",
    "Downloads only `train`+`dev` (~7 GB) and extracts them. XLS-R weights (~1.2 GB) are",
    "pulled from HF automatically on first model build. Training respects `MAX_HOURS`.",
    "Best checkpoint is saved to Drive each time val EER improves.",
))
cells.append(code(
    "if DO_TRAIN:",
    "    fetch(['data/train/*', 'data/dev/*',",
    "           'metadata/train.parquet', 'metadata/dev.parquet'])",
    "    extract_split('train')",
    "    extract_split('dev')",
))
cells.append(code(
    "if DO_TRAIN:",
    "    import subprocess",
    "    cmd = ['python', 'scripts/train_sidq.py',",
    "           '--data-root', 'ArA-DF-2026/data', '--meta-root', 'ArA-DF-2026/metadata',",
    "           '--augmentation', AUGMENTATION, '--epochs', str(EPOCHS),",
    "           '--batch-size', str(BATCH_SIZE), '--patience', str(PATIENCE),",
    "           '--val-subset', str(VAL_SUBSET), '--max-hours', str(MAX_HOURS),",
    "           '--lr', str(LR), '--num-workers', '2', '--attn', 'auto',",
    "           '--output-dir', f'{DRIVE}/checkpoints']",
    "    print(' '.join(cmd))",
    "    subprocess.run(cmd, check=True)",
    "    print('Best checkpoint ->', DRIVE + '/checkpoints/best_sidq.pt')",
))

# ---- Section 7: infer + submit ----
cells.append(md(
    "## 7 · Score the test set & build submission ZIP",
    "Downloads/extracts the chosen `TEST_SPLIT` (if not already present), runs native SIDQ",
    "inference with the trained checkpoint, and writes a **validated** submission ZIP to Drive.",
    "Multi-crop averaging improves robustness for the acoustic-degradation track.",
))
cells.append(code(
    "if DO_INFER_SUBMIT:",
    "    fetch([f'data/{TEST_SPLIT}/*', f'metadata/{TEST_SPLIT}.parquet'])",
    "    extract_split(TEST_SPLIT)",
))
cells.append(code(
    "if DO_INFER_SUBMIT:",
    "    import subprocess",
    "    cmd = ['python', 'scripts/infer_sidq.py',",
    "           '--audio-dir', f'ArA-DF-2026/data/{TEST_SPLIT}',",
    "           '--checkpoint', f'{DRIVE}/checkpoints/best_sidq.pt',",
    "           '--output', '/content/SIDQ/sidq_preds.csv', '--batch-size', '32']",
    "    if MULTI_CROP:",
    "        cmd.append('--multi-crop')",
    "    subprocess.run(cmd, check=True)",
))
cells.append(code(
    "if DO_INFER_SUBMIT:",
    "    import pandas as pd; from pathlib import Path",
    "    from sidq.inference.submission import build_submission, inspect_submission",
    "    preds = pd.read_csv('sidq_preds.csv')",
    "    ref = set(pd.read_parquet(f'ArA-DF-2026/metadata/{TEST_SPLIT}.parquet')['id'])",
    "    tag = f'{AUGMENTATION}_{\"mc\" if MULTI_CROP else \"single\"}'",
    "    out = build_submission(preds, ref, Path(f'{DRIVE}/submission_sidq_{tag}.zip'))",
    "    print('Wrote', out)",
    "    print(inspect_submission(out))",
    "    # Sanity: bonafide scores should trend higher than spoof (check on labeled dev if unsure)",
))

# ---- Section 8: iterate ----
cells.append(md(
    "## 8 · Iterate smartly (stay under budget)",
    "You have **3 submissions** for the eval phase. A budget-aware plan:",
    "",
    "1. **Run 1 — `robust` + multi-crop** (this notebook's default). Bank the ZIP.",
    "2. **Run 2 — `hard` augmentation.** Change `AUGMENTATION='hard'`, rerun §6–7. Heavier",
    "   channel corruptions can help the acoustic-robustness track.",
    "3. **Run 3 — best of the above, or fuse** two checkpoints' scores (average the",
    "   `logit` columns of two `*_preds.csv`, rebuild the ZIP).",
    "",
    "Each training run is only a few compute units, so all three fit comfortably in 100.",
    "",
    "**To save units:** after your final submission, `Runtime ▸ Disconnect and delete runtime`.",
    "An idle connected GPU keeps burning units.",
    "",
    "### Score fusion helper (optional, Run 3)",
))
cells.append(code(
    "# Average two prediction CSVs into a fused submission (often lowers EER).",
    "def fuse(csv_a, csv_b, out_zip, test_split=TEST_SPLIT):",
    "    import pandas as pd; from pathlib import Path",
    "    from scipy.stats import rankdata",
    "    from sidq.inference.submission import build_submission, inspect_submission",
    "    a = pd.read_csv(csv_a).set_index('audio_id')['logit']",
    "    b = pd.read_csv(csv_b).set_index('audio_id')['logit']",
    "    # Rank-average is scale-robust when fusing heterogeneous score ranges",
    "    import numpy as np, pandas as pd",
    "    df = pd.DataFrame({'a': a, 'b': b}).dropna()",
    "    fused = (rankdata(df['a']) + rankdata(df['b'])) / (2*len(df))",
    "    res = pd.DataFrame({'audio_id': df.index, 'logit': fused})",
    "    ref = set(pd.read_parquet(f'ArA-DF-2026/metadata/{test_split}.parquet')['id'])",
    "    out = build_submission(res, ref, Path(out_zip))",
    "    print(inspect_submission(out)); return out",
    "",
    "# Example: fuse(f'{DRIVE}/preds_robust.csv', f'{DRIVE}/preds_hard.csv', f'{DRIVE}/submission_fused.zip')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).parent.parent / "notebooks" / "SIDQ_Colab.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {out_path} with {len(cells)} cells")
