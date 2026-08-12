"""Performance figures for the pennylane_aer results.

Reads only the JSON reports already written by 01/03/04/06 — nothing is re-run,
so this is cheap, deterministic, and safe to regenerate after any of those
scripts is re-executed (on GPU, for instance).

    PYTHONPATH=src python pennylane_aer/05_performance_plots.py

Figures use a log axis wherever the spread is ratio-like (a bar length on a log
axis means nothing, so those become dot plots) and linear bars everywhere else.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
DEEMPHASIS = "#c3c2b7"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
        }
    )


def clean(axes: plt.Axes, value_axis: str = "x") -> None:
    """Hairline grid on the value axis only, no top/right spines."""
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(BASELINE)
    axes.spines["bottom"].set_color(BASELINE)
    axes.grid(True, axis=value_axis, zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(length=0)


def load(name: str) -> dict | None:
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"  skipped: {path.name} not found (run the matching script first)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dot_plot(
    axes: plt.Axes,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    annotations: list[str] | None = None,
) -> None:
    """Horizontal dot plot on a log axis: one row per label, one dot per series."""
    positions = range(len(labels))
    for row in positions:
        values = [values_[row] for _, values_, _ in series]
        axes.plot(
            [min(values), max(values)],
            [row, row],
            color=GRID,
            linewidth=1.5,
            zorder=1,
            solid_capstyle="round",
        )
    for label, values, color in series:
        axes.scatter(
            values,
            list(positions),
            s=70,
            color=color,
            label=label,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
        )
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels)
    axes.set_xscale("log")
    axes.invert_yaxis()
    if annotations:
        limit = axes.get_xlim()[1]
        for row, text in enumerate(annotations):
            if text:
                axes.text(
                    limit,
                    row,
                    text,
                    va="center",
                    ha="right",
                    color=INK_SECONDARY,
                    fontsize=8.5,
                )


def figure_micro_backends(report: dict) -> Path:
    """01 — a 4-qubit circuit across four PennyLane/Aer configurations."""
    results = report["results"]
    labels = [
        r["backend"].replace("qiskit.aer", "Aer").replace(", 4096 shots", "\n4096 shots")
        for r in results
    ]
    forward = [r["forward_seconds"] * 1000 for r in results]
    backward = [r["backward_seconds"] * 1000 for r in results]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9))

    dot_plot(
        left,
        labels,
        [("forward", forward, SERIES_1), ("backward", backward, SERIES_2)],
    )
    clean(left)
    left.set_xlabel("milliseconds per pass (log scale)")
    left.set_title("A single 4-qubit circuit: Aer costs ~19x forward, ~73x backward")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for row, (f_ms, b_ms) in enumerate(zip(forward, backward, strict=True)):
        left.annotate(
            f"{f_ms:,.0f} ms",
            (f_ms, row),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
        left.annotate(
            f"{b_ms:,.0f} ms",
            (b_ms, row),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            color=INK_SECONDARY,
            fontsize=8,
        )

    deviation_rows = [
        ("Aer, default backend", report["default_backend_max_deviation"], "silently sampled"),
        ("Aer, statevector", report["statevector_max_deviation"], "exact"),
        (
            f"Aer, statevector\n{report['shots']} shots",
            report["shot_max_deviation"],
            "~1/sqrt(shots)",
        ),
    ]
    right_labels = [name for name, _, _ in deviation_rows]
    values = [max(value, 1e-17) for _, value, _ in deviation_rows]
    colors = [SERIES_2, SERIES_1, SERIES_2]
    for row, (value, color) in enumerate(zip(values, colors, strict=True)):
        right.plot([1e-17, value], [row, row], color=GRID, linewidth=1.5, zorder=1)
        right.scatter(
            [value],
            [row],
            s=70,
            color=color,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
        )
        right.annotate(
            f"{value:.1e}  {deviation_rows[row][2]}",
            (value, row),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    right.set_yticks(range(len(right_labels)))
    right.set_yticklabels(right_labels)
    right.set_xscale("log")
    right.set_xlim(1e-17, 1e2)
    right.invert_yaxis()
    clean(right)
    right.set_xlabel("max |dp| vs default.qubit (log scale)")
    right.set_title("Only the pinned statevector backend is genuinely exact")

    figure.suptitle(
        "PennyLane on Qiskit Aer, 4 qubits, CPU  |  01_pennylane_aer_circuit.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUTPUT_DIR / "05_perf_micro_backends.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_lambeq_port(report: dict) -> Path:
    """03 — Evelyn's IQP model, same weights, four evaluation backends."""
    results = report["results"]
    labels = [
        r["backend"]
        .replace("tensor-contraction (PytorchQuantumModel)", "tensor contraction\n(baseline)")
        .replace("pennylane-default", "PennyLane\ndefault.qubit")
        .replace("pennylane-aer-1024-shots", "Aer statevector\n1024 shots")
        .replace("pennylane-aer", "Aer statevector\n(exact)")
        for r in results
    ]
    seconds = [r["seconds"] for r in results]
    finite = [bool(r["finite"]) for r in results]
    colors = [SERIES_1 if ok else DEEMPHASIS for ok in finite]

    figure, axes = plt.subplots(figsize=(8.2, 3.8))
    positions = range(len(labels))
    axes.barh(list(positions), seconds, height=0.45, color=colors, zorder=2)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels)
    axes.invert_yaxis()
    clean(axes)
    axes.set_xlim(0, max(seconds) * 1.45)
    axes.set_xlabel(f"seconds for {report['pairs']} sentence pairs")

    for row, record in enumerate(results):
        slowdown = record.get("slowdown_vs_baseline")
        if not record["finite"]:
            note = "NaN logits - predictions invalid"
        elif slowdown is None:
            note = "reference"
        else:
            note = f"{slowdown:.2f}x baseline, predictions identical"
        axes.annotate(
            f"{seconds[row]:.2f} s   {note}",
            (seconds[row], row),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )

    axes.set_title("Aer reproduces the baseline exactly - but only without shots")
    figure.suptitle(
        f"lambeq IQP + cups on Aer, {report['circuit_qubits']} qubits "
        f"({report['post_selected_qubits']} post-selected)  |  03_lambeq_aer_backend.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUTPUT_DIR / "05_perf_lambeq_port.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_cnn_training(report: dict) -> Path:
    """04 — trained configurations: classical control vs hybrid default.qubit."""
    trained = [r for r in report["results"] if r.get("history")]
    if not trained:
        raise ValueError("no trained configuration with a history in the CNN report")

    names = {
        "classical": ("classical control (CNN -> linear)", SERIES_1),
        "hybrid-default.qubit": ("hybrid (CNN -> 4-qubit circuit)", SERIES_2),
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for record in trained:
        label, color = names.get(record["configuration"], (record["configuration"], SERIES_1))
        epochs = [point["epoch"] for point in record["history"]]
        left.plot(
            epochs,
            [point["train_loss"] for point in record["history"]],
            color=color,
            linewidth=2,
            label=label,
        )
        right.plot(
            epochs,
            [point["test_accuracy"] for point in record["history"]],
            color=color,
            linewidth=2,
            label=label,
        )
        right.annotate(
            f"{record['final_test_accuracy']:.3f}",
            (epochs[-1], record["final_test_accuracy"]),
            textcoords="offset points",
            xytext=(6, -2),
            color=color,
            fontsize=8.5,
            fontweight="semibold",
        )

    for axes, ylabel, title in (
        (left, "training loss", "The quantum bottleneck converges slower"),
        (right, "test accuracy", "...and settles ~5 points lower"),
    ):
        clean(axes, value_axis="y")
        axes.set_xlabel("epoch")
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        axes.set_xlim(1, max(point["epoch"] for point in trained[0]["history"]))
    right.set_ylim(0.5, 1.0)
    right.axhline(0.5, color=BASELINE, linewidth=1)
    right.annotate(
        "chance",
        (1.4, 0.508),
        color=INK_MUTED,
        fontsize=8,
    )
    left.legend(loc="upper right")

    dataset = report["dataset"]
    figure.suptitle(
        f"CNN -> quantum hybrid on reduced TREC "
        f"({dataset['train']} train / {dataset['test']} test), "
        f"{report['qubits']} qubits x {report['layers']} layers, seed {report['seed']}"
        "  |  04_cnn_hybrid_quantum.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUTPUT_DIR / "05_perf_cnn_training.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_epoch_cost(report: dict) -> Path:
    """04 — the headline cost of putting a real simulator in the training loop."""
    rows: list[tuple[str, float, str]] = []
    for record in report["results"]:
        seconds = record.get("seconds_per_epoch")
        if seconds is None or not math.isfinite(seconds):
            continue
        if record["configuration"] == "classical":
            rows.append(("classical control", seconds, "measured"))
        elif record["configuration"] == "hybrid-default.qubit":
            rows.append(("hybrid, default.qubit", seconds, "measured"))
        else:
            rows.append(
                (
                    "hybrid, Qiskit Aer",
                    seconds,
                    f"projected from {record.get('steps', '?')} real steps",
                )
            )
    if not rows:
        raise ValueError("no per-epoch timings in the CNN report")

    reference = rows[0][1]
    figure, axes = plt.subplots(figsize=(8.6, 2.7))
    positions = range(len(rows))
    for row, (_, seconds, _) in enumerate(rows):
        axes.plot([reference * 0.5, seconds], [row, row], color=GRID, linewidth=1.5, zorder=1)
    axes.scatter(
        [seconds for _, seconds, _ in rows],
        list(positions),
        s=90,
        color=[SERIES_1, SERIES_1, SERIES_2][: len(rows)],
        zorder=3,
        edgecolors=SURFACE,
        linewidths=2,
    )
    axes.set_yticks(list(positions))
    axes.set_yticklabels([name for name, _, _ in rows])
    axes.set_xscale("log")
    axes.set_xlim(reference * 0.5, rows[-1][1] * 30)
    axes.invert_yaxis()
    clean(axes)
    axes.set_xlabel("seconds per epoch (log scale)")

    for row, (_, seconds, note) in enumerate(rows):
        factor = "" if row == 0 else f"  {seconds / reference:,.0f}x classical"
        axes.annotate(
            f"{seconds:,.2f} s{factor}   ({note})",
            (seconds, row),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )

    axes.set_title("Cost of a real simulator in the training loop")
    if len(rows) == 3:
        figure.text(
            0.005,
            -0.06,
            f"Aer is {rows[2][1] / rows[1][1]:,.0f}x the default.qubit hybrid - "
            "the gap the GPU work has to close",
            color=INK_MUTED,
            fontsize=8.5,
        )
    figure.suptitle(
        "Same model, same data - only the quantum device changes" "  |  04_cnn_hybrid_quantum.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    path = OUTPUT_DIR / "05_perf_epoch_cost.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _interval(summary: dict) -> tuple[float, float, float]:
    """Mean and the two error-bar half-widths, clipped to [0, 1]."""
    mean = summary["mean"]
    low = mean - max(0.0, summary["ci_low"])
    high = min(1.0, summary["ci_high"]) - mean
    return mean, max(0.0, low), max(0.0, high)


def figure_mc1_accuracy(report: dict) -> Path:
    """06 — the quantum layer against both controls, and the scaling ablation."""
    training = report["training"]
    order = ["quantum", "classical", "no-bottleneck"]
    names = {
        "quantum": f"quantum\n({report['configuration']['qubits']}-qubit circuit)",
        "classical": "classical control\n(Linear + tanh)",
        "no-bottleneck": "no bottleneck\n(head only)",
    }
    colors = {"quantum": SERIES_2, "classical": SERIES_1, "no-bottleneck": BASELINE}

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    positions = list(range(len(order)))
    means, lows, highs = [], [], []
    for kind in order:
        mean, low, high = _interval(training[kind]["test_accuracy"])
        means.append(mean)
        lows.append(low)
        highs.append(high)

    left.barh(
        positions,
        means,
        height=0.45,
        color=[colors[k] for k in order],
        zorder=2,
    )
    left.errorbar(
        means,
        positions,
        xerr=[lows, highs],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )
    left.set_yticks(positions)
    left.set_yticklabels([names[k] for k in order])
    left.invert_yaxis()
    clean(left)
    left.set_xlim(0.4, 1.06)
    left.axvline(0.5, color=BASELINE, linewidth=1)
    left.annotate("chance", (0.505, -0.42), color=INK_MUTED, fontsize=8)
    left.set_xlabel(
        f"test accuracy over {report['splits']['test']['pairs']} held-out pairs "
        f"(mean of {len(report['configuration']['seeds'])} seeds, 95% CI)"
    )
    for row, kind in enumerate(order):
        original = training[kind]["test_original_accuracy"]["mean"]
        left.annotate(
            f"{means[row]:.3f}   MC1's own 13 pairs: {original:.3f}",
            (means[row], row),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    left.set_title("The quantum layer matches the controls - all three saturate")

    ablation = report.get("scaling_ablation")
    if ablation:
        labels = ["global\n(one scale for all components)", "per-component\n(unit variance each)"]
        keys = ["global", "per-component"]
        ab_means, ab_lows, ab_highs = [], [], []
        for key in keys:
            mean, low, high = _interval(ablation[key]["test_accuracy"])
            ab_means.append(mean)
            ab_lows.append(low)
            ab_highs.append(high)
        right.barh([0, 1], ab_means, height=0.42, color=[SERIES_1, SERIES_2], zorder=2)
        right.errorbar(
            ab_means,
            [0, 1],
            xerr=[ab_lows, ab_highs],
            fmt="none",
            ecolor=INK_SECONDARY,
            elinewidth=1.2,
            capsize=4,
            zorder=4,
        )
        right.set_yticks([0, 1])
        right.set_yticklabels(labels)
        right.invert_yaxis()
        clean(right)
        right.set_xlim(0.4, 1.06)
        right.axvline(0.5, color=BASELINE, linewidth=1)
        right.set_xlabel("test accuracy (95% CI)")
        for row, key in enumerate(keys):
            train_accuracy = ablation[key].get("train_accuracy", {}).get("mean")
            note = "" if train_accuracy is None else f"   train {train_accuracy:.3f}"
            right.annotate(
                f"{ab_means[row]:.3f}{note}",
                (ab_means[row], row),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                color=INK_SECONDARY,
                fontsize=8.5,
            )
        right.set_title("Angle scaling decides the result")
        figure.text(
            0.56,
            -0.04,
            "Per-component scaling lifts the low-variance components - syntax, not topic - to\n"
            "the amplitude of the topic-carrying one: the circuit sees noise at signal strength.",
            color=INK_MUTED,
            fontsize=8.5,
        )

    configuration = report["configuration"]
    figure.suptitle(
        f"MC1 through {configuration['embedding']} + {configuration['reducer']} "
        f"-> {configuration['qubits']} qubits "
        f"({configuration['layers']} layers, {configuration['reuploads']} re-uploads, "
        f"0 post-selected)  |  06_mc1_reduced.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUTPUT_DIR / "05_perf_mc1_accuracy.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_mc1_width(report: dict) -> Path:
    """06 — accuracy against circuit width: where the two layers separate."""
    sweep = report["qubit_sweep"]
    widths = sorted(int(k) for k in sweep)

    figure, axes = plt.subplots(figsize=(8.6, 4.0))
    for kind, color, label in (
        ("quantum", SERIES_2, "quantum layer"),
        ("classical", SERIES_1, "classical control"),
    ):
        means, lows, highs = [], [], []
        for width in widths:
            mean, low, high = _interval(sweep[str(width)][kind]["test_accuracy"])
            means.append(mean)
            lows.append(low)
            highs.append(high)
        axes.errorbar(
            widths,
            means,
            yerr=[lows, highs],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            capsize=4,
            elinewidth=1.1,
            label=label,
            zorder=3,
        )

    clean(axes, value_axis="y")
    axes.set_xticks(widths)
    axes.set_xlabel("qubits (= retained components)")
    axes.set_ylabel("test accuracy")
    axes.set_ylim(0.4, 1.04)
    axes.axhline(0.5, color=BASELINE, linewidth=1)
    axes.annotate("chance", (widths[0], 0.508), color=INK_MUTED, fontsize=8)
    axes.legend(loc="lower right")

    variance = axes.twinx()
    ratios = [sweep[str(w)].get("explained_variance_ratio") for w in widths]
    if all(r is not None for r in ratios):
        variance.plot(
            widths,
            ratios,
            color=INK_MUTED,
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        variance.set_ylabel("explained variance retained", color=INK_MUTED)
        variance.set_ylim(0, 1.04)
        variance.tick_params(length=0)
        for side in ("top", "left", "bottom"):
            variance.spines[side].set_visible(False)
        variance.spines["right"].set_color(BASELINE)
        variance.annotate(
            "explained variance",
            (widths[-1], ratios[-1]),
            textcoords="offset points",
            xytext=(-6, 8),
            ha="right",
            color=INK_MUTED,
            fontsize=8.5,
        )

    axes.set_title("The quantum layer degrades as width grows; the control does not")
    figure.text(
        0.005,
        -0.03,
        "Explained variance rises with width while quantum accuracy falls, so this is "
        "optimisation, not representation:\nthe extra circuit parameters memorise the 58 "
        "training sentences. 2 qubits is the right width for MC1.",
        color=INK_MUTED,
        fontsize=8.5,
    )
    figure.suptitle(
        f"MC1, circuit width sweep, {len(report['configuration']['seeds'])} seeds per point"
        "  |  06_mc1_reduced.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUTPUT_DIR / "05_perf_mc1_width.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_mc1_aer(report: dict) -> Path:
    """06 — Aer with shots works here, and SPSA is what makes it trainable."""
    evaluation = report["aer_evaluation"]
    training = report.get("aer_training", {})
    benchmarks = training.get("benchmarks", [])

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9))

    labels = []
    for record in evaluation["results"]:
        name = record["backend"].replace("aer-", "Aer ").replace("-", " ")
        labels.append(name)
    accuracies = [r["test_accuracy"] for r in evaluation["results"]]
    agreements = [r["prediction_agreement"] for r in evaluation["results"]]
    positions = list(range(len(labels)))

    left.barh(
        [p - 0.19 for p in positions],
        accuracies,
        height=0.34,
        color=SERIES_1,
        label="test accuracy",
        zorder=2,
    )
    left.barh(
        [p + 0.19 for p in positions],
        agreements,
        height=0.34,
        color=SERIES_2,
        label="prediction agreement\nwith default.qubit",
        zorder=2,
    )
    left.axvline(
        evaluation["reference_accuracy"],
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    left.set_yticks(positions)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    clean(left)
    left.set_xlim(0, 1.18)
    left.set_xlabel(
        f"{evaluation['pairs']} held-out pairs "
        f"from {evaluation['sentences']} sentences ({evaluation['circuit_qubits']} qubits, "
        f"{evaluation['post_selected_qubits']} post-selected)"
    )
    for row, record in enumerate(evaluation["results"]):
        left.annotate(
            f"{record['test_accuracy']:.3f}",
            (record["test_accuracy"], row - 0.19),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
        left.annotate(
            f"{record['prediction_agreement']:.3f}   max |dlogit| "
            f"{record['max_abs_logit_difference']:.1e}",
            (record["prediction_agreement"], row + 0.19),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    left.set_title("Shots work here - the lambeq route returned NaN")
    figure.text(
        0.005,
        -0.04,
        "Sampling moves the logits by ~1 at 1024 shots, yet every prediction still matches the\n"
        "exact run: the decision boundary is far enough from the data to survive shot noise.",
        color=INK_MUTED,
        fontsize=8.5,
    )

    if benchmarks:
        order = sorted(benchmarks, key=lambda b: b["projected_seconds_per_epoch"])
        names = [b["diff_method"] for b in order]
        seconds = [b["projected_seconds_per_epoch"] for b in order]
        rows = list(range(len(order)))
        floor = min(seconds) * 0.35
        for row, value in zip(rows, seconds, strict=True):
            right.plot([floor, value], [row, row], color=GRID, linewidth=1.5, zorder=1)
        right.scatter(
            seconds,
            rows,
            s=90,
            color=[SERIES_1, SERIES_2][: len(order)],
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
        )
        right.set_yticks(rows)
        right.set_yticklabels(names)
        right.set_xscale("log")
        right.set_xlim(floor, max(seconds) * 12)
        right.invert_yaxis()
        clean(right)
        right.set_xlabel("projected seconds per epoch on Aer (log scale)")
        for row, record in enumerate(order):
            right.annotate(
                f"{record['projected_seconds_per_epoch'] / 60:,.1f} min   "
                f"{record['circuit_evaluations_per_sentence']} circuit evals/sentence",
                (record["projected_seconds_per_epoch"], row),
                textcoords="offset points",
                xytext=(10, 0),
                va="center",
                color=INK_SECONDARY,
                fontsize=8.5,
            )
        if len(order) == 2:
            right.set_title(
                f"SPSA makes Aer trainable: {seconds[1] / seconds[0]:.0f}x cheaper per epoch"
            )
        else:
            right.set_title("Aer training cost")
        figure.text(
            0.56,
            -0.04,
            f"One epoch covers all {report['splits']['train']['pairs']:,} training pairs but costs "
            f"{training.get('sentences_per_epoch', '?')} circuit evaluations -\n"
            "pair features are indexed from per-sentence outputs, so cost tracks sentences.",
            color=INK_MUTED,
            fontsize=8.5,
        )

    figure.suptitle(
        "MC1 reduced route on Qiskit Aer, weights transferred from default.qubit"
        "  |  06_mc1_reduced.json",
        x=0.005,
        ha="left",
        color=INK_MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUTPUT_DIR / "05_perf_mc1_aer.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    micro = load("01_pennylane_aer_circuit.json")
    if micro:
        written.append(figure_micro_backends(micro))

    port = load("03_lambeq_aer_backend.json")
    if port:
        written.append(figure_lambeq_port(port))

    hybrid = load("04_cnn_hybrid_quantum.json")
    if hybrid:
        written.append(figure_cnn_training(hybrid))
        written.append(figure_epoch_cost(hybrid))

    reduced = load("06_mc1_reduced.json")
    if reduced:
        written.append(figure_mc1_accuracy(reduced))
        if reduced.get("qubit_sweep"):
            written.append(figure_mc1_width(reduced))
        if reduced.get("aer_evaluation"):
            written.append(figure_mc1_aer(reduced))

    for path in written:
        print(f"Wrote {path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
