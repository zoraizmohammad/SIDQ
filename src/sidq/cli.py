"""SIDQ command-line interface."""

import typer

app = typer.Typer(
    name="sidq",
    help="SIDQ-T2: Arabic Speech Deepfake Detection under Degraded Quality",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print SIDQ version."""
    from sidq import __version__

    typer.echo(f"sidq {__version__}")


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to experiment config YAML"),
) -> None:
    """Train a model from configuration."""
    typer.echo(f"Training with config: {config}")
    raise typer.Exit(code=1)


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
    config: str = typer.Option("", help="Evaluation config override"),
) -> None:
    """Evaluate a trained model."""
    typer.echo(f"Evaluating checkpoint: {checkpoint}")
    raise typer.Exit(code=1)


@app.command()
def infer(
    checkpoint: str = typer.Argument(..., help="Model checkpoint or ensemble config"),
    output: str = typer.Option("predictions.parquet", help="Output path"),
) -> None:
    """Run inference and produce predictions."""
    typer.echo(f"Inference from: {checkpoint} -> {output}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
