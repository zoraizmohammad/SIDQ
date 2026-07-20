"""AWS stress-test loop: Polly generation + corruption + scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sidq.aws.config import AWSConfig


@dataclass
class StressTestConfig:
    """Configuration for the AWS stress-test loop."""

    prompt_manifest: Path | None = None
    polly_voices: list[str] = field(default_factory=lambda: ["Zeina"])
    max_generations: int = 10
    budget_usd: float = 1.0
    dry_run: bool = True
    output_dir: Path = Path("artifacts/aws_stress")


@dataclass
class StressTestReport:
    """Report from stress-test loop execution."""

    num_generated: int = 0
    num_scored: int = 0
    num_transcribed: int = 0
    cost_spent_usd: float = 0.0
    fragile_conditions: list[str] = field(default_factory=list)
    suggested_augmentation: str = ""
    dry_run: bool = True


def run_stress_test(
    config: StressTestConfig,
    aws_config: AWSConfig,
    model_checkpoint: Path | None = None,
) -> StressTestReport:
    """Run the AWS stress-test loop.

    Steps:
    1. Validate user opt-in (config.dry_run or aws_config.enabled)
    2. Estimate total cost
    3. Generate controlled spoof audio via Polly
    4. Apply corruption ladder to generated audio
    5. Score every variant with the SIDQ model
    6. Optionally transcribe variants
    7. Compute score drift and rank fragile conditions
    8. Produce report with suggested augmentation config

    This function NEVER:
    - Retrains models automatically
    - Submits results to Codabench
    - Uploads official competition audio to AWS
    - Spends money without explicit budget configuration
    """
    report = StressTestReport(dry_run=config.dry_run)

    if config.dry_run:
        report.suggested_augmentation = "robust"
        return report

    aws_config.validate_enabled()
    aws_config.cost_guard.check(config.budget_usd)

    # The full implementation would:
    # 1. Load prompts from manifest
    # 2. Generate via Polly with cost tracking
    # 3. Apply codec/noise/reverb ladder
    # 4. Score with loaded model
    # 5. Rank conditions by score drift
    # 6. Suggest augmentation focus

    raise NotImplementedError(
        "Full stress-test loop requires model checkpoint loading and "
        "live AWS credentials. Use dry_run=True for pipeline testing."
    )
