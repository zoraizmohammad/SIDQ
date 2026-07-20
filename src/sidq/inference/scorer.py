"""Inference scoring pipeline for generating predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sidq.constants import SAMPLE_RATE


class AudioInferenceDataset(Dataset):
    """Simple dataset for inference over audio files."""

    def __init__(
        self,
        audio_ids: list[str],
        audio_paths: list[Path],
        crop_duration_sec: float | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.audio_ids = audio_ids
        self.audio_paths = audio_paths
        self.crop_duration_sec = crop_duration_sec
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.audio_ids)

    def __getitem__(self, idx: int) -> dict[str, any]:
        from sidq.audio.io import load_audio

        audio_id = self.audio_ids[idx]
        path = self.audio_paths[idx]

        waveform, sr = load_audio(
            path, target_sr=self.sample_rate, max_duration_sec=20.0
        )

        if self.crop_duration_sec is not None:
            target_samples = int(self.crop_duration_sec * self.sample_rate)
            if waveform.shape[-1] > target_samples:
                waveform = waveform[:target_samples]
            elif waveform.shape[-1] < target_samples:
                pad = target_samples - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))

        return {"audio_id": audio_id, "waveform": waveform}


def collate_variable_length(batch: list[dict]) -> dict:
    """Collate variable-length waveforms with padding."""
    audio_ids = [item["audio_id"] for item in batch]
    waveforms = [item["waveform"] for item in batch]

    max_len = max(w.shape[-1] for w in waveforms)
    lengths = torch.tensor([w.shape[-1] for w in waveforms])

    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[-1]] = w

    return {"audio_ids": audio_ids, "waveforms": padded, "lengths": lengths}


@torch.no_grad()
def score_batch(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    lengths: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Score a batch of waveforms, returning bonafide logits."""
    if device is None:
        device = next(model.parameters()).device

    waveforms = waveforms.to(device)
    if lengths is not None:
        lengths = lengths.to(device)

    model.eval()
    logits = model(waveforms, lengths=lengths)

    # Return bonafide-class score (index 1)
    bonafide_scores = logits[:, 1].cpu().numpy()
    return bonafide_scores


def run_inference(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run full inference over a dataset.

    Returns DataFrame with columns: audio_id, logit
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        pin_memory=torch.cuda.is_available(),
    )

    all_ids: list[str] = []
    all_scores: list[float] = []

    for batch in loader:
        scores = score_batch(
            model, batch["waveforms"], batch["lengths"], device=device
        )
        all_ids.extend(batch["audio_ids"])
        all_scores.extend(scores.tolist())

    df = pd.DataFrame({"audio_id": all_ids, "logit": all_scores})

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
