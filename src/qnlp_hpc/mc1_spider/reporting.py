from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from qnlp_hpc.mc1_spider.config import OUTPUT_DIR

def to_float_list(values: Sequence[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            result.append(float(value.detach().cpu().item()))
        else:
            result.append(float(value))
    return result


def save_benchmark_plot(
    values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Path,
    best_epoch: int,
) -> None:
    epochs = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, values)

    if 1 <= best_epoch <= len(values):
        best_value = values[best_epoch - 1]
        ax.axvline(best_epoch, linestyle="--", alpha=0.45)
        ax.scatter([best_epoch], [best_value], facecolors="none", edgecolors="black")
        ax.annotate(
            f"best epoch = {best_epoch}",
            xy=(best_epoch, best_value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel == "Accuracy":
        ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_four_benchmark_plots(
    history: pd.DataFrame,
    best_epoch: int,
) -> None:
    save_benchmark_plot(
        history["train_loss"].tolist(),
        ylabel="Loss",
        title="Training set — Loss",
        output_path=OUTPUT_DIR / "benchmark_train_loss.png",
        best_epoch=best_epoch,
    )
    save_benchmark_plot(
        history["development_loss"].tolist(),
        ylabel="Loss",
        title="Development set — Loss",
        output_path=OUTPUT_DIR / "benchmark_development_loss.png",
        best_epoch=best_epoch,
    )
    save_benchmark_plot(
        history["train_accuracy"].tolist(),
        ylabel="Accuracy",
        title="Training set — Accuracy",
        output_path=OUTPUT_DIR / "benchmark_train_accuracy.png",
        best_epoch=best_epoch,
    )
    save_benchmark_plot(
        history["development_accuracy"].tolist(),
        ylabel="Accuracy",
        title="Development set — Accuracy",
        output_path=OUTPUT_DIR / "benchmark_development_accuracy.png",
        best_epoch=best_epoch,
    )


def get_best_epoch(history: pd.DataFrame) -> int:
    ranked = history.sort_values(
        ["development_loss", "development_accuracy", "epoch"],
        ascending=[True, False, True],
    )
    return int(ranked.iloc[0]["epoch"])