"""Main entry point for CLI when running as module."""

import sys
from pathlib import Path

import typer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.eval import app as eval_app
from src.cli.train import app as train_app

app = typer.Typer(
    name="wmtp",
    help="WMTP Fine-Tuning Framework CLI",
    pretty_exceptions_show_locals=False,
)

app.add_typer(train_app, name="train", help="Train models")
app.add_typer(eval_app, name="eval", help="Evaluate models")


@app.callback()
def main():
    """
    WMTP Fine-Tuning Framework - Meta Multi-Token Prediction

    Train and evaluate models using critic-weighted or Rho-1 weighted
    Multi-Token Prediction approaches.
    """
    pass


if __name__ == "__main__":
    app()
