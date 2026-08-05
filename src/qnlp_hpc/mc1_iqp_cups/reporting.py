"""Convert training histories and save benchmark plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import SupportsFloat, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from qnlp_hpc.mc1_iqp_cups.config import OUTPUT_DIR


def to_float_list(values: Sequence[object]) -> list[float]:
    """Convert a mixed tensor/numeric metric history to Python floats."""
    result: list[float] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            result.append(float(value.detach().cpu().item()))
        else:
            result.append(float(cast(SupportsFloat, value)))
    return result


def save_benchmark_plot(
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Path,
    selected_epoch: int,
) -> None:
    """Save one metric history and mark the selected epoch."""
    epochs = np.arange(1, len(values) + 1)
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(epochs, values)

    if 1 <= selected_epoch <= len(values):
        selected_value = values[selected_epoch - 1]
        axes.axvline(selected_epoch, linestyle="--", alpha=0.45)
        axes.scatter(
            [selected_epoch],
            [selected_value],
            facecolors="none",
            edgecolors="black",
        )
        axes.annotate(
            f"selected epoch = {selected_epoch}\n(min dev loss)",
            xy=(selected_epoch, selected_value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )

    axes.set_xlabel("Epochs")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if ylabel == "Accuracy":
        axes.set_ylim(0.0, 1.02)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_four_benchmark_plots(
    history: pd.DataFrame,
    selected_epoch: int,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save training/development loss and accuracy plots."""
    plot_specs = (
        (
            "train_loss",
            "Loss",
            "Training set — Loss",
            "benchmark_train_loss.png",
        ),
        (
            "development_loss",
            "Loss",
            "Development set — Loss",
            "benchmark_development_loss.png",
        ),
        (
            "train_accuracy",
            "Accuracy",
            "Training set — Accuracy",
            "benchmark_train_accuracy.png",
        ),
        (
            "development_accuracy",
            "Accuracy",
            "Development set — Accuracy",
            "benchmark_development_accuracy.png",
        ),
    )

    for column, ylabel, title, filename in plot_specs:
        save_benchmark_plot(
            history[column].tolist(),
            ylabel=ylabel,
            title=title,
            output_path=output_dir / filename,
            selected_epoch=selected_epoch,
        )


def get_best_epoch(history: pd.DataFrame) -> int:
    """Select minimum development loss, breaking ties deterministically."""
    ranked = history.sort_values(
        ["development_loss", "development_accuracy", "epoch"],
        ascending=[True, False, True],
    )
    return int(ranked.iloc[0]["epoch"])
