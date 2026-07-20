"""Amazon Polly integration for controlled spoof generation. Optional."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sidq.aws.config import AWSConfig, get_boto3_client


@dataclass
class PollyRequest:
    """A single Polly synthesis request."""

    text: str
    voice_id: str = "Zeina"
    engine: str = "standard"
    language_code: str = "arb"
    output_format: str = "pcm"
    sample_rate: str = "16000"
    ssml: bool = False

    @property
    def content_hash(self) -> str:
        content = f"{self.text}|{self.voice_id}|{self.engine}|{self.sample_rate}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class PollyResult:
    """Result from a Polly synthesis."""

    audio_path: Path
    request: PollyRequest
    cost_estimate_usd: float = 0.0


def estimate_polly_cost(text: str, engine: str = "standard") -> float:
    """Estimate cost for a single synthesis request.

    Standard voices: $4 per 1M characters.
    Neural voices: $16 per 1M characters.
    """
    n_chars = len(text)
    if engine == "neural":
        return n_chars * 16.0 / 1_000_000
    return n_chars * 4.0 / 1_000_000


def synthesize(
    request: PollyRequest,
    config: AWSConfig,
    output_dir: Path,
) -> PollyResult:
    """Synthesize speech via Amazon Polly.

    Requires explicit opt-in via config.enabled and cost budget.
    """
    config.validate_enabled()

    cost = estimate_polly_cost(request.text, request.engine)
    config.cost_guard.check(cost)

    output_path = output_dir / f"polly_{request.content_hash}.wav"
    if output_path.exists():
        return PollyResult(audio_path=output_path, request=request, cost_estimate_usd=0.0)

    if config.dry_run:
        return PollyResult(audio_path=output_path, request=request, cost_estimate_usd=cost)

    client = get_boto3_client("polly", config)

    text_type = "ssml" if request.ssml else "text"
    response = client.synthesize_speech(
        Text=request.text,
        TextType=text_type,
        VoiceId=request.voice_id,
        Engine=request.engine,
        LanguageCode=request.language_code,
        OutputFormat=request.output_format,
        SampleRate=request.sample_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_stream = response["AudioStream"].read()
    output_path.write_bytes(audio_stream)

    config.cost_guard.record(cost)

    return PollyResult(audio_path=output_path, request=request, cost_estimate_usd=cost)


def list_arabic_voices(config: AWSConfig) -> list[dict]:
    """List available Arabic voices."""
    config.validate_enabled()
    if config.dry_run:
        return [
            {"Id": "Zeina", "LanguageCode": "arb", "Engine": "standard"},
        ]
    client = get_boto3_client("polly", config)
    response = client.describe_voices(LanguageCode="arb")
    return response.get("Voices", [])
