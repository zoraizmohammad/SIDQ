"""Codec augmentation via ffmpeg subprocess."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sidq.augment.base import AudioTransform, TransformMetadata

CODEC_CONFIGS = {
    "mp3_64k": {"codec": "libmp3lame", "bitrate": "64k", "ext": "mp3"},
    "mp3_128k": {"codec": "libmp3lame", "bitrate": "128k", "ext": "mp3"},
    "aac_64k": {"codec": "aac", "bitrate": "64k", "ext": "aac"},
    "aac_128k": {"codec": "aac", "bitrate": "128k", "ext": "aac"},
    "opus_32k": {"codec": "libopus", "bitrate": "32k", "ext": "opus"},
    "opus_64k": {"codec": "libopus", "bitrate": "64k", "ext": "opus"},
    "ogg_64k": {"codec": "libvorbis", "bitrate": "64k", "ext": "ogg"},
    "g711_ulaw": {"codec": "pcm_mulaw", "bitrate": None, "ext": "wav", "ar": "8000"},
    "g711_alaw": {"codec": "pcm_alaw", "bitrate": None, "ext": "wav", "ar": "8000"},
}


class CodecTransform(AudioTransform):
    """Apply codec compression/decompression via ffmpeg."""

    def __init__(
        self,
        codecs: list[str] | None = None,
        double_encode: bool = False,
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)
        self.codecs = codecs or list(CODEC_CONFIGS.keys())
        self.double_encode = double_encode

    @property
    def name(self) -> str:
        return "codec"

    @property
    def external_requirements(self) -> list[str]:
        return ["ffmpeg"]

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        codec_name = rng.choice(self.codecs)
        config = CODEC_CONFIGS[codec_name]

        result = self._encode_decode(waveform, sample_rate, config)

        if self.double_encode and result is not None:
            second_codec = rng.choice(self.codecs)
            second_config = CODEC_CONFIGS[second_codec]
            result2 = self._encode_decode(result, sample_rate, second_config)
            if result2 is not None:
                result = result2
                codec_name = f"{codec_name}+{second_codec}"

        if result is None:
            return waveform, TransformMetadata(
                transform_name=f"{self.name}:failed", parameters={"codec": codec_name}
            )

        return result, TransformMetadata(
            transform_name=self.name,
            parameters={"codec": codec_name, "double": self.double_encode},
        )

    def _encode_decode(
        self, waveform: torch.Tensor, sample_rate: int, config: dict
    ) -> torch.Tensor | None:
        """Encode and decode waveform through a codec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            output_path = Path(tmpdir) / f"output.{config['ext']}"
            decoded_path = Path(tmpdir) / "decoded.wav"

            audio_np = waveform.numpy()
            sf.write(str(input_path), audio_np, sample_rate)

            cmd = ["ffmpeg", "-y", "-i", str(input_path)]
            if config.get("ar"):
                cmd.extend(["-ar", config["ar"]])
            cmd.extend(["-acodec", config["codec"]])
            if config["bitrate"]:
                cmd.extend(["-b:a", config["bitrate"]])
            cmd.append(str(output_path))

            try:
                subprocess.run(
                    cmd, capture_output=True, check=True, timeout=30
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None

            decode_cmd = [
                "ffmpeg", "-y", "-i", str(output_path),
                "-ar", str(sample_rate), "-ac", "1",
                str(decoded_path),
            ]
            try:
                subprocess.run(
                    decode_cmd, capture_output=True, check=True, timeout=30
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return None

            if not decoded_path.exists():
                return None

            data, _ = sf.read(str(decoded_path))
            result = torch.from_numpy(data).float()

            target_len = waveform.shape[-1]
            if result.shape[-1] > target_len:
                result = result[:target_len]
            elif result.shape[-1] < target_len:
                result = torch.nn.functional.pad(result, (0, target_len - result.shape[-1]))

            return result


def check_ffmpeg_codecs() -> dict[str, bool]:
    """Check which codecs are available via ffmpeg."""
    if not shutil.which("ffmpeg"):
        return {k: False for k in CODEC_CONFIGS}

    try:
        result = subprocess.run(
            ["ffmpeg", "-codecs"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {k: False for k in CODEC_CONFIGS}

    available = {}
    for name, config in CODEC_CONFIGS.items():
        codec = config["codec"]
        available[name] = codec in output
    return available
