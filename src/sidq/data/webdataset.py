"""WebDataset streaming loader for TAR shards."""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sidq.constants import SAMPLE_RATE


class WebDatasetLoader(IterableDataset):
    """Stream audio samples from WebDataset TAR shards.

    Expects shards containing .flac audio files and .json metadata sidecars.
    """

    def __init__(
        self,
        shard_urls: list[str],
        sample_rate: int = SAMPLE_RATE,
        max_duration_sec: float = 20.0,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_urls = shard_urls
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.shuffle_shards = shuffle_shards
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples from TAR shards."""
        try:
            import webdataset as wds
        except ImportError as e:
            raise ImportError(
                "webdataset is required for streaming mode. "
                "Install with: pip install webdataset"
            ) from e

        worker_info = torch.utils.data.get_worker_info()
        urls = self.shard_urls

        if worker_info is not None:
            urls = urls[worker_info.id :: worker_info.num_workers]

        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed)
            urls = [urls[i] for i in rng.permutation(len(urls))]

        dataset = wds.WebDataset(urls).decode("pil")

        for sample in dataset:
            try:
                result = self._process_sample(sample)
                if result is not None:
                    yield result
            except Exception:
                continue

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single WebDataset sample."""
        key = sample.get("__key__", "")

        audio_data = None
        for ext in (".flac", ".wav", ".mp3"):
            if ext.lstrip(".") in sample:
                audio_data = sample[ext.lstrip(".")]
                break

        if audio_data is None:
            return None

        waveform = self._decode_audio(audio_data)
        if waveform is None:
            return None

        max_samples = int(self.max_duration_sec * self.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

        metadata = {}
        if "json" in sample:
            metadata = sample["json"] if isinstance(sample["json"], dict) else {}

        return {
            "audio_id": key,
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "label": metadata.get("label"),
            "metadata": metadata,
        }

    def _decode_audio(self, audio_data: Any) -> torch.Tensor | None:
        """Decode audio bytes to tensor."""
        try:
            import soundfile as sf

            if isinstance(audio_data, (bytes, bytearray)):
                data, sr = sf.read(io.BytesIO(audio_data))
            else:
                return None

            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=-1)

            if sr != self.sample_rate:
                import torchaudio

                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0), sr, self.sample_rate
                ).squeeze(0)

            if not torch.isfinite(waveform).all():
                return None

            return waveform
        except Exception:
            return None
