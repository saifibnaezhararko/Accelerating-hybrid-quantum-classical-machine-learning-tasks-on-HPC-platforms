"""MC1 + dimensionality reduction on PennyLane and Qiskit Aer.

Runs the reduced-embedding route against Evelyn's sentence-disjoint MC1 split:

  * multi-seed training on ``default.qubit`` with two classical controls,
  * an ablation of the angle-scaling choice, which is what decides the result,
  * an optional circuit-width sweep over the 2-8 qubit range,
  * transfer of the trained weights onto Qiskit Aer - exact *and shot-based*,
    the latter being impossible on the lambeq route (README trap 2),
  * measured Aer training cost, parameter-shift versus SPSA.

Run from the repo root:  python pennylane_aer/06_train_mc1_reduced.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_ROOT / "src"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aer_backend import aer_backend_config, aer_gpu_available, describe_backend
from mc1_reduced import (
    ClassicalPairClassifier,
    NoBottleneckPairClassifier,
    QuantumPairClassifier,
    SentenceAngleEncoder,
    augmented_pairs,
    build_split,
    derive_sentence_topics,
    evaluate_split,
    frame_pairs,
    mean_confidence_interval,
    train_pair_model,
)

from qnlp_hpc.mc1_iqp_cups import config as evelyn_config
from qnlp_hpc.mc1_iqp_cups import data as evelyn_data

OUTPUT_DIR = _HERE / "outputs"
REPORT_PATH = OUTPUT_DIR / "06_mc1_reduced.json"

ALL_STAGES = frozenset({"training", "scaling", "augmentation", "width", "aer-eval", "aer-training"})


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def prepare_data(arguments, seed: int, augment_train: bool = True):
    """Build the four evaluation splits for one seed.

    Split membership comes from ``mc1_iqp_cups.config``, so this shares
    Evelyn's sentence-disjoint boundary exactly and the numbers stay
    comparable to the lambeq baseline.

    ``augment_train=False`` trains on the 64 pairs MC1 ships for the training
    split instead of all 1,653 pairs its training sentences can form.  The
    evaluation splits never change, so the two are directly comparable.
    """
    frame = evelyn_data.load_mc1(evelyn_config.DATA_PATH)
    topics = derive_sentence_topics(frame)

    all_sentences = set(frame["sentence_1"]).union(frame["sentence_2"])
    test_sentences = sorted(evelyn_config.TEST_SENTENCES)
    development_sentences = sorted(evelyn_config.DEVELOPMENT_SENTENCES)
    train_sentences = sorted(
        all_sentences - evelyn_config.TEST_SENTENCES - evelyn_config.DEVELOPMENT_SENTENCES
    )

    encoder = SentenceAngleEncoder(
        n_qubits=arguments.qubits,
        embedding=arguments.embedding,
        reducer=arguments.reducer,
        scaling=arguments.scaling,
        seed=seed,
        bert_model=arguments.bert_model,
    ).fit(train_sentences)

    train_frame, _, test_frame, _ = evelyn_data.split_sentence_disjoint(frame)
    train_pairs = (
        augmented_pairs(train_sentences, topics) if augment_train else frame_pairs(train_frame)
    )

    splits = {
        "train": build_split("train", train_pairs, encoder, train_sentences),
        "development": build_split(
            "development",
            augmented_pairs(development_sentences, topics),
            encoder,
            development_sentences,
        ),
        "test": build_split(
            "test", augmented_pairs(test_sentences, topics), encoder, test_sentences
        ),
        # The 13 pairs MC1 actually ships for the test split: the exact set the
        # lambeq baseline reported 1.0000 on.
        "test_original": build_split("test_original", frame_pairs(test_frame), encoder),
    }
    return splits, encoder, topics


def build_model(kind: str, arguments, backend_config=None, diff_method: str = "best"):
    if kind == "quantum":
        return QuantumPairClassifier(
            n_qubits=arguments.qubits,
            n_layers=arguments.layers,
            reuploads=arguments.reuploads,
            hidden_dim=evelyn_config.CLASSIFIER_HIDDEN_DIM,
            dropout=evelyn_config.CLASSIFIER_DROPOUT,
            backend_config=backend_config,
            diff_method=diff_method,
        )
    if kind == "classical":
        return ClassicalPairClassifier(
            arguments.qubits,
            evelyn_config.CLASSIFIER_HIDDEN_DIM,
            evelyn_config.CLASSIFIER_DROPOUT,
        )
    if kind == "no-bottleneck":
        return NoBottleneckPairClassifier(
            arguments.qubits,
            evelyn_config.CLASSIFIER_HIDDEN_DIM,
            evelyn_config.CLASSIFIER_DROPOUT,
        )
    raise ValueError(f"Unknown model kind {kind!r}.")


def run_one(kind: str, arguments, splits, seed: int, verbose: bool = False):
    set_seed(seed)
    model = build_model(kind, arguments)
    record = train_pair_model(
        model,
        splits["train"],
        splits["development"],
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        verbose_every=max(1, arguments.epochs // 6) if verbose else 0,
    )
    for name, split in splits.items():
        accuracy, loss = evaluate_split(model, split)
        record[f"{name}_accuracy"] = accuracy
        record[f"{name}_loss"] = loss
    record["model"] = kind
    record["seed"] = seed
    record["parameters"] = sum(p.numel() for p in model.parameters())
    record["quantum_parameters"] = sum(
        p.numel() for n, p in model.named_parameters() if "quantum" in n
    )
    if not arguments.keep_history:
        record.pop("history", None)
    return model, record


def summarise(records: list[dict], key: str) -> dict:
    return mean_confidence_interval([r[key] for r in records])


def format_interval(summary: dict) -> str:
    """Student-t interval, clipped to [0, 1] because accuracy is bounded."""
    return (
        f"{summary['mean']:.4f} "
        f"[{max(0.0, summary['ci_low']):.4f}, {min(1.0, summary['ci_high']):.4f}]"
    )


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_training(arguments, seeds: list[int]):
    """Multi-seed training of the quantum model and both classical controls."""
    print("\n=== 1. multi-seed training on default.qubit ===")
    per_model: dict[str, list[dict]] = {}
    best_model = None
    best_accuracy = -1.0

    for kind in ("quantum", "classical", "no-bottleneck"):
        per_model[kind] = []
        print(f"\n  [{kind}]")
        for seed in seeds:
            splits, _, _ = prepare_data(arguments, seed)
            model, record = run_one(kind, arguments, splits, seed, verbose=arguments.verbose)
            per_model[kind].append(record)
            print(
                f"    seed {seed}: dev={record['development_accuracy']:.3f}  "
                f"test={record['test_accuracy']:.3f} ({len(splits['test'])} pairs)  "
                f"test_original={record['test_original_accuracy']:.3f} "
                f"({len(splits['test_original'])} pairs)  "
                f"epoch {record['selected_epoch']}  {record['seconds']:.1f}s"
            )
            if kind == "quantum" and record["test_accuracy"] > best_accuracy:
                best_accuracy = record["test_accuracy"]
                best_model = (model, seed, splits)

    print(f"\n  {'model':16s} {'test acc (95% CI)':>28s} {'MC1 13 pairs':>14s} {'s/epoch':>9s}")
    summary = {}
    for kind, records in per_model.items():
        test = summarise(records, "test_accuracy")
        original = summarise(records, "test_original_accuracy")
        epoch = summarise(records, "seconds_per_epoch")
        summary[kind] = {
            "test_accuracy": test,
            "test_original_accuracy": original,
            "development_accuracy": summarise(records, "development_accuracy"),
            "seconds_per_epoch": epoch,
            "parameters": records[0]["parameters"],
            "quantum_parameters": records[0]["quantum_parameters"],
            "runs": records,
        }
        print(
            f"  {kind:16s} {format_interval(test):>28s} "
            f"{original['mean']:14.4f} {epoch['mean']:9.3f}"
        )

    return summary, best_model


def stage_scaling_ablation(arguments, seeds: list[int]) -> dict:
    """Global versus per-component angle scaling - the decisive design choice."""
    print("\n=== 2. angle-scaling ablation ===")
    results = {}
    original_scaling = arguments.scaling
    for scaling in ("global", "per-component"):
        arguments.scaling = scaling
        records = []
        for seed in seeds:
            splits, encoder, _ = prepare_data(arguments, seed)
            _, record = run_one("quantum", arguments, splits, seed)
            records.append(record)
        test = summarise(records, "test_accuracy")
        train = summarise(records, "train_accuracy")
        results[scaling] = {"test_accuracy": test, "train_accuracy": train, "runs": records}
        print(f"  {scaling:16s} test={format_interval(test)}  train={train['mean']:.4f}")
    arguments.scaling = original_scaling
    return results


def stage_augmentation_ablation(arguments, seeds: list[int]) -> dict:
    """MC1's 64 shipped training pairs against all 1,653 its sentences allow.

    Topic labels are derived from MC1's own labels and pairs are formed only
    within the training sentence set, so this adds supervision without
    crossing the sentence-disjoint boundary.
    """
    print("\n=== 3. training-set augmentation ablation ===")
    results = {}
    for augment in (False, True):
        records = []
        pairs = 0
        for seed in seeds:
            splits, _, _ = prepare_data(arguments, seed, augment_train=augment)
            pairs = len(splits["train"])
            _, record = run_one("quantum", arguments, splits, seed)
            records.append(record)
        test = summarise(records, "test_accuracy")
        original = summarise(records, "test_original_accuracy")
        label = "augmented" if augment else "shipped pairs"
        results[label] = {
            "train_pairs": pairs,
            "test_accuracy": test,
            "test_original_accuracy": original,
            "runs": records,
        }
        print(
            f"  {label:16s} {pairs:5d} train pairs  test={format_interval(test)}  "
            f"MC1 13 pairs={original['mean']:.4f}"
        )
    return results


def stage_qubit_sweep(arguments, seeds: list[int], widths: list[int]) -> dict:
    """Accuracy against circuit width, quantum layer versus classical control.

    The control is swept too: at the widths where both saturate, the
    comparison says nothing, and only the narrow end separates them.
    """
    print("\n=== 4. circuit-width sweep ===")
    results = {}
    original_qubits = arguments.qubits
    print(
        f"  {'qubits':>7s} {'evr':>7s} {'q-params':>9s} "
        f"{'quantum test (95% CI)':>28s} {'classical test (95% CI)':>28s}"
    )
    for width in widths:
        arguments.qubits = width
        by_kind: dict[str, list[dict]] = {"quantum": [], "classical": []}
        explained = None
        for seed in seeds:
            splits, encoder, _ = prepare_data(arguments, seed)
            explained = encoder.explained_variance_ratio
            for kind in by_kind:
                _, record = run_one(kind, arguments, splits, seed)
                by_kind[kind].append(record)

        quantum = summarise(by_kind["quantum"], "test_accuracy")
        classical = summarise(by_kind["classical"], "test_accuracy")
        results[str(width)] = {
            "explained_variance_ratio": explained,
            "quantum_parameters": by_kind["quantum"][0]["quantum_parameters"],
            "quantum": {
                "test_accuracy": quantum,
                "development_accuracy": summarise(by_kind["quantum"], "development_accuracy"),
                "seconds_per_epoch": summarise(by_kind["quantum"], "seconds_per_epoch"),
            },
            "classical": {
                "test_accuracy": classical,
                "development_accuracy": summarise(by_kind["classical"], "development_accuracy"),
            },
        }
        evr = "n/a" if explained is None else f"{explained:.3f}"
        print(
            f"  {width:7d} {evr:>7s} {by_kind['quantum'][0]['quantum_parameters']:9d} "
            f"{format_interval(quantum):>28s} {format_interval(classical):>28s}"
        )
    arguments.qubits = original_qubits
    return results


def stage_aer_evaluation(arguments, trained_model, splits, shot_counts: list[int]) -> dict:
    """Transfer trained weights onto Aer and evaluate, exact and with shots."""
    print("\n=== 5. Qiskit Aer evaluation (weights transferred from default.qubit) ===")
    if arguments.gpu and not aer_gpu_available():
        print("  --gpu requested but this Aer build exposes no GPU device; using CPU.")

    split = splits["test"]
    original = splits["test_original"]
    trained_model.eval()
    with torch.no_grad():
        reference_logits = trained_model(split.angles, split.left, split.right)
    reference_predictions = reference_logits.argmax(dim=1)
    reference_accuracy = float((reference_predictions == split.labels).float().mean())
    print(
        f"  reference (default.qubit): test={reference_accuracy:.4f} on {len(split)} pairs "
        f"from {len(split.sentences)} sentences"
    )

    configurations: list[tuple[str, dict]] = [("aer-exact", aer_backend_config(gpu=arguments.gpu))]
    for shots in shot_counts:
        configurations.append(
            (f"aer-{shots}-shots", aer_backend_config(gpu=arguments.gpu, shots=shots))
        )

    results = []
    for label, backend_config in configurations:
        model = build_model("quantum", arguments, backend_config=backend_config)
        model.load_state_dict(trained_model.state_dict())
        model.eval()

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(split.angles, split.left, split.right)
            original_logits = model(original.angles, original.left, original.right)
        seconds = time.perf_counter() - start

        finite = bool(torch.isfinite(logits).all())
        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == split.labels).float().mean())
        original_accuracy = float((original_logits.argmax(dim=1) == original.labels).float().mean())
        agreement = float((predictions == reference_predictions).float().mean())
        difference = float((logits - reference_logits).abs().max()) if finite else float("nan")

        print(
            f"  {label:18s} {describe_backend(backend_config)}\n"
            f"      {seconds:6.2f}s  finite={finite}  test={accuracy:.4f}  "
            f"MC1 13 pairs={original_accuracy:.4f}  "
            f"agreement={agreement:.4f}  max|dlogit|={difference:.2e}"
        )
        results.append(
            {
                "backend": label,
                "backend_config": describe_backend(backend_config),
                "seconds": seconds,
                "finite": finite,
                "test_accuracy": accuracy,
                "test_original_accuracy": original_accuracy,
                "prediction_agreement": agreement,
                "max_abs_logit_difference": difference,
            }
        )

    return {
        "reference_accuracy": reference_accuracy,
        "pairs": len(split),
        "sentences": len(split.sentences),
        "circuit_qubits": arguments.qubits,
        "post_selected_qubits": 0,
        "results": results,
    }


def stage_aer_training(arguments, splits, methods: list[str]) -> dict:
    """Measure - and optionally complete - a training run on Aer.

    Cost here is per *distinct sentence*, not per pair, so one epoch costs the
    same whether it covers 13 pairs or all 1653.
    """
    print("\n=== 6. Aer training cost ===")
    backend_config = aer_backend_config(gpu=arguments.gpu)
    train = splits["train"]
    n_sentences = len(train.sentences)
    results = []

    for method in methods:
        set_seed(arguments.seeds_list[0])
        model = build_model("quantum", arguments, backend_config=backend_config, diff_method=method)
        optimiser = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
        quantum_parameters = sum(p.numel() for n, p in model.named_parameters() if "quantum" in n)

        # One measured step over a subset, then project to a full epoch.
        subset = torch.arange(min(arguments.aer_probe_sentences, n_sentences))
        mask = torch.isin(train.left, subset) & torch.isin(train.right, subset)
        pairs = int(mask.sum())
        # Project on the sentences the circuit actually ran, not on the subset
        # size: that is what the cost is proportional to.
        probe = int(torch.unique(torch.cat((train.left[mask], train.right[mask]))).numel())
        if probe == 0:
            raise RuntimeError(
                "The Aer probe subset produced no pairs; raise --aer-probe-sentences."
            )

        model.train()
        start = time.perf_counter()
        optimiser.zero_grad()
        logits = model(train.angles, train.left[mask], train.right[mask])
        loss = torch.nn.functional.cross_entropy(logits, train.labels[mask])
        loss.backward()
        optimiser.step()
        seconds = time.perf_counter() - start

        gradient = sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if "quantum" in n and p.grad is not None
        )
        per_sentence = seconds / probe
        projected_epoch = per_sentence * n_sentences

        print(
            f"  {method:16s} {seconds:6.2f}s for {probe} sentences ({pairs} pairs)  "
            f"-> {per_sentence:5.2f}s/sentence, {projected_epoch / 60:5.2f} min/epoch "
            f"(projected, {n_sentences} sentences, all {len(train)} pairs)"
        )
        print(f"                   gradient reaching circuit weights: {gradient:.6f}")

        results.append(
            {
                "diff_method": method,
                "probe_sentences": probe,
                "probe_pairs": pairs,
                "measured_seconds": seconds,
                "seconds_per_sentence": per_sentence,
                "projected_seconds_per_epoch": projected_epoch,
                "quantum_parameters": quantum_parameters,
                "quantum_gradient_norm": gradient,
                "circuit_evaluations_per_sentence": (
                    2 if method == "spsa" else 2 * quantum_parameters
                ),
            }
        )

    trained = None
    if arguments.aer_epochs > 0:
        method = methods[0]
        print(f"\n  training on Aer for {arguments.aer_epochs} epochs with diff_method={method}")
        set_seed(arguments.seeds_list[0])
        model = build_model("quantum", arguments, backend_config=backend_config, diff_method=method)
        record = train_pair_model(
            model,
            train,
            splits["development"],
            epochs=arguments.aer_epochs,
            batch_size=len(train),
            learning_rate=arguments.aer_learning_rate,
            patience=None,
            verbose_every=1,
        )
        for name, split in splits.items():
            accuracy, _ = evaluate_split(model, split)
            record[f"{name}_accuracy"] = accuracy
        record["diff_method"] = method
        if not arguments.keep_history:
            record.pop("history", None)
        print(
            f"    trained on Aer: test={record['test_accuracy']:.4f}  "
            f"MC1 13 pairs={record['test_original_accuracy']:.4f}  "
            f"{record['seconds_per_epoch']:.1f}s/epoch"
        )
        trained = record

    return {"sentences_per_epoch": n_sentences, "benchmarks": results, "trained": trained}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:
            start, end = chunk.rsplit("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise argparse.ArgumentTypeError(f"No seeds parsed from {text!r}.")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4, help="Circuit width after reduction.")
    parser.add_argument("--layers", type=int, default=2, help="StronglyEntanglingLayers depth.")
    parser.add_argument("--reuploads", type=int, default=2, help="Angle re-upload blocks.")
    parser.add_argument("--embedding", default="tfidf", choices=("count", "tfidf", "bert"))
    parser.add_argument("--reducer", default="pca", choices=("pca", "tsvd", "umap"))
    parser.add_argument("--scaling", default="global", choices=("global", "per-component"))
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--seeds", type=parse_seeds, default="0-4")
    parser.add_argument("--keep-history", action="store_true", help="Keep per-epoch curves.")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--qubit-sweep", default="", help="e.g. 2,3,4,6,8 (empty to skip).")
    parser.add_argument("--skip-aer", action="store_true")
    parser.add_argument("--aer-shots", default="1024,8192", help="Comma-separated, empty to skip.")
    parser.add_argument("--aer-diff", default="spsa,parameter-shift")
    parser.add_argument("--aer-probe-sentences", type=int, default=4)
    parser.add_argument("--aer-epochs", type=int, default=0, help="Real Aer training epochs.")
    parser.add_argument(
        "--aer-learning-rate",
        type=float,
        default=5e-2,
        # SPSA estimates the gradient from two evaluations, so it needs a
        # smaller step than exact backprop: measured on default.qubit at 2
        # qubits, SPSA reaches 1.000 at 5e-2 but only 0.909 at 2e-1, while
        # backprop reaches 1.000 at both.
        help="Step size for the Aer training run; SPSA needs a smaller one than backprop.",
    )
    parser.add_argument(
        "--only",
        default="",
        help=(
            "Comma-separated subset of stages to run, merged into the existing "
            f"report: {','.join(sorted(ALL_STAGES))}."
        ),
    )
    parser.add_argument("--gpu", action="store_true", help="Use Aer's GPU statevector.")
    arguments = parser.parse_args()

    seeds = arguments.seeds if isinstance(arguments.seeds, list) else parse_seeds(arguments.seeds)
    arguments.seeds_list = seeds

    if arguments.only:
        requested = {stage.strip() for stage in arguments.only.split(",") if stage.strip()}
        unknown = requested - ALL_STAGES
        if unknown:
            parser.error(f"Unknown stage(s) {sorted(unknown)}; choose from {sorted(ALL_STAGES)}.")
    else:
        requested = set(ALL_STAGES)
        if arguments.skip_ablation:
            requested -= {"scaling", "augmentation"}
        if arguments.skip_aer:
            requested -= {"aer-eval", "aer-training"}
        if not arguments.qubit_sweep:
            requested -= {"width"}

    splits, encoder, topics = prepare_data(arguments, seeds[0])
    print("MC1 through dimensionality reduction")
    print(
        f"  embedding={arguments.embedding}  reducer={arguments.reducer}  "
        f"scaling={arguments.scaling}  qubits={arguments.qubits}  "
        f"layers={arguments.layers}  reuploads={arguments.reuploads}"
    )
    if encoder.explained_variance_ratio is not None:
        print(f"  explained variance retained: {encoder.explained_variance_ratio:.3f}")
    print(f"  derived sentence topics: {len(topics)} sentences, 2 topics (0 label disagreements)")
    for name, split in splits.items():
        print(
            f"  {name:14s} {len(split.sentences):3d} sentences  {len(split):5d} pairs  "
            f"labels {split.label_counts}"
        )
    print(f"  seeds: {seeds}")

    report: dict[str, object] = {
        "configuration": {
            "qubits": arguments.qubits,
            "layers": arguments.layers,
            "reuploads": arguments.reuploads,
            "embedding": arguments.embedding,
            "reducer": arguments.reducer,
            "scaling": arguments.scaling,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "seeds": seeds,
            "explained_variance_ratio": encoder.explained_variance_ratio,
        },
        "splits": {
            name: {
                "sentences": len(split.sentences),
                "pairs": len(split),
                "labels": split.label_counts,
            }
            for name, split in splits.items()
        },
        "aer_gpu_available": aer_gpu_available(),
    }

    # --only merges into the existing report instead of replacing it, so a
    # single stage can be re-run - on a GPU node, for instance - without
    # spending an hour reproducing the stages that did not change.
    if requested != ALL_STAGES and REPORT_PATH.exists():
        report = {**json.loads(REPORT_PATH.read_text(encoding="utf-8")), **report}
        print(f"  merging into the existing {REPORT_PATH.name}")

    best = None
    if "training" in requested:
        training, best = stage_training(arguments, seeds)
        report["training"] = training

    if "scaling" in requested:
        report["scaling_ablation"] = stage_scaling_ablation(arguments, seeds)
    if "augmentation" in requested:
        report["augmentation_ablation"] = stage_augmentation_ablation(arguments, seeds)

    if "width" in requested and arguments.qubit_sweep:
        widths = [int(w) for w in arguments.qubit_sweep.split(",") if w.strip()]
        report["qubit_sweep"] = stage_qubit_sweep(arguments, seeds, widths)

    if requested & {"aer-eval", "aer-training"}:
        if best is None:
            # The Aer stages need trained weights to transfer; when stage 1 was
            # not requested, train the first seed just for that.
            print(f"\n  training seed {seeds[0]} on default.qubit for the Aer stages")
            reference_splits, _, _ = prepare_data(arguments, seeds[0])
            model, record = run_one("quantum", arguments, reference_splits, seeds[0])
            print(f"    reference test accuracy: {record['test_accuracy']:.4f}")
            best = (model, seeds[0], reference_splits)

        trained_model, _, best_splits = best
        if "aer-eval" in requested:
            shot_counts = [int(s) for s in arguments.aer_shots.split(",") if s.strip()]
            report["aer_evaluation"] = stage_aer_evaluation(
                arguments, trained_model, best_splits, shot_counts
            )
        if "aer-training" in requested:
            methods = [m.strip() for m in arguments.aer_diff.split(",") if m.strip()]
            if methods:
                report["aer_training"] = stage_aer_training(arguments, best_splits, methods)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
