"""AWS configuration and cost guards. All AWS use is optional and opt-in."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostGuard:
    """Budget enforcement for AWS operations.

    Tracks estimated spend and refuses operations that exceed the budget.
    """

    budget_usd: float = 0.0
    spent_usd: float = 0.0
    enabled: bool = True

    def check(self, estimated_cost_usd: float) -> None:
        """Raise if the operation would exceed the budget."""
        if not self.enabled:
            raise RuntimeError(
                "CostGuard is disabled. AWS operations require an explicit "
                "budget. Set budget_usd and enabled=True."
            )
        if self.budget_usd <= 0:
            raise RuntimeError(
                "No budget configured. Set budget_usd to a positive value "
                "to authorize AWS spending."
            )
        if self.spent_usd + estimated_cost_usd > self.budget_usd:
            raise RuntimeError(
                f"Operation cost ${estimated_cost_usd:.4f} would exceed "
                f"budget: spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f}."
            )

    def record(self, cost_usd: float) -> None:
        """Record actual spend."""
        self.spent_usd += cost_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)


@dataclass
class AWSConfig:
    """AWS integration configuration. Disabled by default."""

    enabled: bool = False
    region: str = "us-east-1"
    profile: str | None = None
    s3_bucket: str | None = None
    dry_run: bool = True
    cost_guard: CostGuard = field(default_factory=CostGuard)

    def validate_enabled(self) -> None:
        """Raise unless AWS use has been explicitly enabled."""
        if not self.enabled:
            raise RuntimeError(
                "AWS integration is disabled by default. Set enabled=True "
                "in the AWS config to opt in. No AWS calls will be made "
                "without explicit opt-in."
            )


def get_boto3_client(service: str, config: AWSConfig):
    """Create a boto3 client using the standard credential chain.

    Never reads hard-coded credentials. Uses environment, shared
    credentials file, or instance profile via the standard chain.
    """
    config.validate_enabled()
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "boto3 is required for AWS integration. "
            "Install with: pip install 'sidq[aws]'"
        ) from e

    session_kwargs = {"region_name": config.region}
    if config.profile:
        session_kwargs["profile_name"] = config.profile

    session = boto3.Session(**session_kwargs)
    return session.client(service)
