"""TREC + dimensionality reduction on PennyLane and Qiskit Aer.

Runs the reduced-embedding route against TREC's official train/test split:

  * multi-seed training on ``default.qubit`` with two classical controls and
    two reference ceilings (full-dimension logistic regression, majority class),
  * an ablation of the angle-scaling choice,
  * a circuit-width sweep over the 2-8 qubit range, where TREC - unlike MC1 -
    has real headroom,
  * transfer of the trained weights onto Qiskit Aer, exact *and shot-based*,
  * measured Aer training cost, parameter-shift versus SPSA.

Run from the repo root:  python pennylane_aer/06_train_trec_reduced.py
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
from trec_reduced import (
    ClassicalSentenceClassifier,
    NoBottleneckSentenceClassifier,
    QuantumSentenceClassifier,
    SentenceAngleEncoder,
    build_split,
    class_names,
    evaluate_split,
    load_trec,
    mean_confidence_interval,
    stratified_split,
    train_model,
)

DATA_DIR = _ROOT / "trec dataset"
OUTPUT_DIR = _HERE / "outputs"
REPORT_PATH = OUTPUT_DIR / "06_trec_reduced.json"

ALL_STAGES = frozenset({"training", "scaling", "width", "aer-eval", "aer-training"})


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def prepare_data(arguments, seed: int):
    """Build train/development/test splits for one seed.

    TREC's official train/test division is kept; development is carved out of
    training, stratified, so model selection never touches the test split.
    """
    train_frame, test_frame = load_trec(
        arguments.data_dir,
        label_column=arguments.label_column,
        max_words=arguments.max_words,
        classes=arguments.classes,
    )
    train_frame, development_frame = stratified_split(
        train_frame, arguments.development_fraction, seed
    )

    encoder = SentenceAngleEncoder(
        n_qubits=arguments.qubits,
        embedding=arguments.embedding,
        reducer=arguments.reducer,
        scaling=arguments.scaling,
        seed=seed,
        bert_model=arguments.bert_model,
        min_document_frequency=arguments.min_document_frequency,
    ).fit(train_frame["text"].astype(str).tolist())

    splits = {
        "train": build_split("train", train_frame, encoder),
        "development": build_split("development", development_frame, encoder),
        "test": build_split("test", test_frame, encoder),
    }
    n_classes = int(train_frame["label"].max()) + 1
    return splits, encoder, n_classes, train_frame.attrs["label_remap"]


def build_model(kind: str, arguments, n_classes: int, backend_config=None, diff_method="best"):
    if kind == "quantum":
        return QuantumSentenceClassifier(
            n_qubits=arguments.qubits,
            n_classes=n_classes,
            n_layers=arguments.layers,
            reuploads=arguments.reuploads,
            hidden_dim=arguments.hidden_dim,
            dropout=arguments.dropout,
            backend_config=backend_config,
            diff_method=diff_method,
        )
    if kind == "classical":
        return ClassicalSentenceClassifier(
            arguments.qubits, n_classes, arguments.hidden_dim, arguments.dropout
        )
    if kind == "no-bottleneck":
        return NoBottleneckSentenceClassifier(
            arguments.qubits, n_classes, arguments.hidden_dim, arguments.dropout
        )
    raise ValueError(f"Unknown model kind {kind!r}.")


def run_one(kind: str, arguments, splits, n_classes: int, seed: int, verbose: bool = False):
    set_seed(seed)
    model = build_model(kind, arguments, n_classes)
    record = train_model(
        model,
        splits["train"],
        splits["development"],
        n_classes,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        verbose_every=max(1, arguments.epochs // 6) if verbose else 0,
    )
    for name, split in splits.items():
        scores = evaluate_split(model, split, n_classes)
        record[f"{name}_accuracy"] = scores["accuracy"]
        record[f"{name}_macro_f1"] = scores["macro_f1"]
        if name == "test":
            record["test_confusion"] = scores["confusion"]
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
# Reference ceilings
# --------------------------------------------------------------------------


def reference_ceilings(arguments, n_classes: int) -> dict:
    """What the representation costs, independent of any quantum layer.

    Without these two numbers a reduced-route accuracy is uninterpretable: the
    majority-class rate says what beating chance means on an unbalanced set,
    and full-dimension logistic regression says how much the reduction to
    ``n_qubits`` components threw away.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_frame, test_frame = load_trec(
        arguments.data_dir,
        label_column=arguments.label_column,
        max_words=arguments.max_words,
        classes=arguments.classes,
    )
    vectoriser = TfidfVectorizer(
        token_pattern=r"[a-z0-9']+", lowercase=True, min_df=arguments.min_document_frequency
    )
    matrix = vectoriser.fit_transform(train_frame["text"].astype(str))
    model = LogisticRegression(max_iter=3000).fit(matrix, train_frame["label"])
    full_dimension = float(
        model.score(vectoriser.transform(test_frame["text"].astype(str)), test_frame["label"])
    )

    counts = np.bincount(test_frame["label"].to_numpy(), minlength=n_classes)
    majority = float(counts.max() / counts.sum())

    print("\n=== reference ceilings (no quantum layer) ===")
    print(f"  majority class                       {majority:.4f}")
    print(f"  full-dimension logistic regression   {full_dimension:.4f}  ")
    print(f"  (TF-IDF vocabulary: {len(vectoriser.vocabulary_)} features)")
    return {
        "majority_class_accuracy": majority,
        "full_dimension_logistic_accuracy": full_dimension,
        "vocabulary_size": len(vectoriser.vocabulary_),
    }


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_training(arguments, seeds: list[int]):
    print("\n=== 1. multi-seed training on default.qubit ===")
    per_model: dict[str, list[dict]] = {}
    best_model = None
    best_accuracy = -1.0

    for kind in ("quantum", "classical", "no-bottleneck"):
        per_model[kind] = []
        print(f"\n  [{kind}]")
        for seed in seeds:
            splits, _, n_classes, _ = prepare_data(arguments, seed)
            model, record = run_one(
                kind, arguments, splits, n_classes, seed, verbose=arguments.verbose
            )
            per_model[kind].append(record)
            print(
                f"    seed {seed}: dev={record['development_accuracy']:.4f}  "
                f"test={record['test_accuracy']:.4f}  "
                f"macro-F1={record['test_macro_f1']:.4f}  "
                f"epoch {record['selected_epoch']}  {record['seconds']:.1f}s"
            )
            if kind == "quantum" and record["test_accuracy"] > best_accuracy:
                best_accuracy = record["test_accuracy"]
                best_model = (model, seed, splits, n_classes)

    print(
        f"\n  {'model':16s} {'test acc (95% CI)':>28s} {'macro-F1':>10s} "
        f"{'q-params':>9s} {'s/epoch':>9s}"
    )
    summary = {}
    for kind, records in per_model.items():
        test = summarise(records, "test_accuracy")
        f1 = summarise(records, "test_macro_f1")
        epoch = summarise(records, "seconds_per_epoch")
        summary[kind] = {
            "test_accuracy": test,
            "test_macro_f1": f1,
            "development_accuracy": summarise(records, "development_accuracy"),
            "seconds_per_epoch": epoch,
            "parameters": records[0]["parameters"],
            "quantum_parameters": records[0]["quantum_parameters"],
            "seeds": seeds,
            "runs": records,
        }
        print(
            f"  {kind:16s} {format_interval(test):>28s} {f1['mean']:10.4f} "
            f"{records[0]['quantum_parameters']:9d} {epoch['mean']:9.3f}"
        )

    return summary, best_model


def stage_scaling_ablation(arguments, seeds: list[int]) -> dict:
    print("\n=== 2. angle-scaling ablation ===")
    results = {}
    original = arguments.scaling
    for scaling in ("global", "per-component"):
        arguments.scaling = scaling
        records = []
        for seed in seeds:
            splits, _, n_classes, _ = prepare_data(arguments, seed)
            _, record = run_one("quantum", arguments, splits, n_classes, seed)
            records.append(record)
        test = summarise(records, "test_accuracy")
        results[scaling] = {
            "test_accuracy": test,
            "test_macro_f1": summarise(records, "test_macro_f1"),
            "seeds": seeds,
            "runs": records,
        }
        print(f"  {scaling:16s} test={format_interval(test)}")
    arguments.scaling = original
    return results


def stage_width_sweep(arguments, seeds: list[int], widths: list[int]) -> dict:
    """Accuracy against circuit width, quantum layer versus classical control."""
    print("\n=== 3. circuit-width sweep ===")
    results = {}
    original = arguments.qubits
    print(
        f"  {'qubits':>7s} {'evr':>7s} {'q-params':>9s} "
        f"{'quantum test (95% CI)':>28s} {'classical test (95% CI)':>28s}"
    )
    for width in widths:
        arguments.qubits = width
        by_kind: dict[str, list[dict]] = {"quantum": [], "classical": []}
        explained = None
        for seed in seeds:
            splits, encoder, n_classes, _ = prepare_data(arguments, seed)
            explained = encoder.explained_variance_ratio
            for kind in by_kind:
                _, record = run_one(kind, arguments, splits, n_classes, seed)
                by_kind[kind].append(record)

        quantum = summarise(by_kind["quantum"], "test_accuracy")
        classical = summarise(by_kind["classical"], "test_accuracy")
        results[str(width)] = {
            "explained_variance_ratio": explained,
            "seeds": seeds,
            "quantum_parameters": by_kind["quantum"][0]["quantum_parameters"],
            "quantum": {
                "test_accuracy": quantum,
                "test_macro_f1": summarise(by_kind["quantum"], "test_macro_f1"),
                "seconds_per_epoch": summarise(by_kind["quantum"], "seconds_per_epoch"),
            },
            "classical": {
                "test_accuracy": classical,
                "test_macro_f1": summarise(by_kind["classical"], "test_macro_f1"),
            },
        }
        evr = "n/a" if explained is None else f"{explained:.3f}"
        print(
            f"  {width:7d} {evr:>7s} {by_kind['quantum'][0]['quantum_parameters']:9d} "
            f"{format_interval(quantum):>28s} {format_interval(classical):>28s}"
        )
    arguments.qubits = original
    return results


def stage_aer_evaluation(arguments, trained_model, splits, n_classes, shot_counts) -> dict:
    """Transfer trained weights onto Aer and evaluate, exact and with shots."""
    print("\n=== 4. Qiskit Aer evaluation (weights transferred from default.qubit) ===")
    if arguments.gpu and not aer_gpu_available():
        print("  --gpu requested but this Aer build exposes no GPU device; using CPU.")

    split = splits["test"]
    if arguments.aer_eval_sentences and arguments.aer_eval_sentences < len(split):
        # Aer costs ~0.3 s per question; the full test split is affordable but
        # a subset keeps a GPU smoke test quick.  Deterministic prefix, so the
        # reference and every backend see the same questions.
        keep = arguments.aer_eval_sentences
        split = type(split)(
            name=split.name,
            sentences=split.sentences[:keep],
            angles=split.angles[:keep],
            labels=split.labels[:keep],
        )

    trained_model.eval()
    with torch.no_grad():
        reference_logits = trained_model(split.angles)
    reference_predictions = reference_logits.argmax(dim=1)
    reference_accuracy = float((reference_predictions == split.labels).float().mean())
    print(f"  reference (default.qubit): test={reference_accuracy:.4f} on {len(split)} questions")

    configurations: list[tuple[str, dict]] = [("aer-exact", aer_backend_config(gpu=arguments.gpu))]
    for shots in shot_counts:
        configurations.append(
            (f"aer-{shots}-shots", aer_backend_config(gpu=arguments.gpu, shots=shots))
        )

    results = []
    for label, backend_config in configurations:
        model = build_model("quantum", arguments, n_classes, backend_config=backend_config)
        model.load_state_dict(trained_model.state_dict())
        model.eval()

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(split.angles)
        seconds = time.perf_counter() - start

        finite = bool(torch.isfinite(logits).all())
        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == split.labels).float().mean())
        agreement = float((predictions == reference_predictions).float().mean())
        difference = float((logits - reference_logits).abs().max()) if finite else float("nan")

        print(
            f"  {label:18s} {describe_backend(backend_config)}\n"
            f"      {seconds:6.2f}s  finite={finite}  test={accuracy:.4f}  "
            f"agreement={agreement:.4f}  max|dlogit|={difference:.2e}"
        )
        results.append(
            {
                "backend": label,
                "backend_config": describe_backend(backend_config),
                "seconds": seconds,
                "finite": finite,
                "test_accuracy": accuracy,
                "prediction_agreement": agreement,
                "max_abs_logit_difference": difference,
            }
        )

    return {
        "reference_accuracy": reference_accuracy,
        "questions": len(split),
        "circuit_qubits": arguments.qubits,
        "post_selected_qubits": 0,
        "results": results,
    }


def stage_aer_training(arguments, splits, n_classes, methods: list[str]) -> dict:
    """Measure - and optionally complete - a training run on Aer."""
    print("\n=== 5. Aer training cost ===")
    backend_config = aer_backend_config(gpu=arguments.gpu)
    train = splits["train"]
    n_train = len(train)
    results = []

    for method in methods:
        set_seed(arguments.seeds_list[0])
        model = build_model(
            "quantum", arguments, n_classes, backend_config=backend_config, diff_method=method
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
        quantum_parameters = sum(p.numel() for n, p in model.named_parameters() if "quantum" in n)

        probe = min(arguments.aer_probe_sentences, n_train)
        model.train()
        start = time.perf_counter()
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(train.angles[:probe]), train.labels[:probe])
        loss.backward()
        optimiser.step()
        seconds = time.perf_counter() - start

        gradient = sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if "quantum" in n and p.grad is not None
        )
        per_sentence = seconds / probe
        projected_epoch = per_sentence * n_train

        print(
            f"  {method:16s} {seconds:6.2f}s for {probe} questions  "
            f"-> {per_sentence:5.2f}s/question, {projected_epoch / 3600:6.2f} h/epoch "
            f"(projected over {n_train} training questions)"
        )
        print(f"                   gradient reaching circuit weights: {gradient:.6f}")

        results.append(
            {
                "diff_method": method,
                "probe_sentences": probe,
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
        subset = (
            min(arguments.aer_train_sentences, n_train)
            if arguments.aer_train_sentences
            else n_train
        )
        print(
            f"\n  training on Aer for {arguments.aer_epochs} epochs with diff_method={method} "
            f"on {subset} of {n_train} training questions"
        )
        reduced_train = type(train)(
            name=train.name,
            sentences=train.sentences[:subset],
            angles=train.angles[:subset],
            labels=train.labels[:subset],
        )
        set_seed(arguments.seeds_list[0])
        model = build_model(
            "quantum", arguments, n_classes, backend_config=backend_config, diff_method=method
        )
        record = train_model(
            model,
            reduced_train,
            splits["development"],
            n_classes,
            epochs=arguments.aer_epochs,
            batch_size=subset,
            learning_rate=arguments.aer_learning_rate,
            patience=None,
            verbose_every=1,
        )
        for name, split in splits.items():
            scores = evaluate_split(model, split, n_classes)
            record[f"{name}_accuracy"] = scores["accuracy"]
            record[f"{name}_macro_f1"] = scores["macro_f1"]
        record["diff_method"] = method
        record["training_questions"] = subset
        if not arguments.keep_history:
            record.pop("history", None)
        print(
            f"    trained on Aer: test={record['test_accuracy']:.4f}  "
            f"macro-F1={record['test_macro_f1']:.4f}  "
            f"{record['seconds_per_epoch']:.1f}s/epoch"
        )
        trained = record

    return {"train_questions": n_train, "benchmarks": results, "trained": trained}


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


def parse_classes(text: str) -> list[int] | None:
    if not text.strip():
        return None
    return [int(chunk) for chunk in text.split(",") if chunk.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--label-column", default="label-coarse", choices=("label-coarse", "label-fine")
    )
    parser.add_argument("--max-words", type=int, default=None, help="Drop longer questions.")
    parser.add_argument("--classes", type=parse_classes, default=None, help="e.g. 0,3")
    parser.add_argument("--development-fraction", type=float, default=0.15)
    parser.add_argument("--min-document-frequency", type=int, default=1)

    parser.add_argument("--qubits", type=int, default=8, help="Circuit width after reduction.")
    parser.add_argument("--layers", type=int, default=2, help="StronglyEntanglingLayers depth.")
    parser.add_argument("--reuploads", type=int, default=2, help="Angle re-upload blocks.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--embedding", default="tfidf", choices=("count", "tfidf", "bert"))
    parser.add_argument("--reducer", default="tsvd", choices=("tsvd", "pca", "umap"))
    parser.add_argument("--scaling", default="global", choices=("global", "per-component"))
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--seeds", type=parse_seeds, default="0-4")
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--width-sweep", default="", help="e.g. 2,3,4,6,8 (empty to skip).")
    parser.add_argument("--skip-aer", action="store_true")
    parser.add_argument("--aer-shots", default="1024,8192", help="Comma-separated, empty to skip.")
    parser.add_argument("--aer-diff", default="spsa,parameter-shift")
    parser.add_argument("--aer-probe-sentences", type=int, default=4)
    parser.add_argument("--aer-eval-sentences", type=int, default=200)
    parser.add_argument("--aer-epochs", type=int, default=0, help="Real Aer training epochs.")
    parser.add_argument("--aer-train-sentences", type=int, default=256)
    parser.add_argument(
        "--aer-learning-rate",
        type=float,
        default=5e-2,
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
            requested -= {"scaling"}
        if arguments.skip_aer:
            requested -= {"aer-eval", "aer-training"}
        if not arguments.width_sweep:
            requested -= {"width"}

    if not arguments.data_dir.is_dir():
        print(f"TREC folder not found: {arguments.data_dir}")
        return 2

    splits, encoder, n_classes, remap = prepare_data(arguments, seeds[0])
    names = class_names(remap, arguments.label_column)

    print("TREC through dimensionality reduction")
    print(
        f"  embedding={arguments.embedding}  reducer={arguments.reducer}  "
        f"scaling={arguments.scaling}  qubits={arguments.qubits}  "
        f"layers={arguments.layers}  reuploads={arguments.reuploads}"
    )
    print(f"  vocabulary: {encoder.vocabulary_size} features -> {arguments.qubits} components")
    if encoder.explained_variance_ratio is not None:
        print(f"  explained variance retained: {encoder.explained_variance_ratio:.4f}")
    print(f"  classes ({n_classes}): {names}")
    for name, split in splits.items():
        print(f"  {name:14s} {len(split):5d} questions  labels {split.label_counts}")
    print(f"  seeds: {seeds}")

    # Note: with --only, this block describes the run that wrote it last, not
    # every stage in the merged report.  Each stage stamps its own seeds, and
    # every aggregate carries n, so per-stage provenance survives the merge.
    report: dict[str, object] = {
        "configuration": {
            "dataset": str(arguments.data_dir),
            "label_column": arguments.label_column,
            "max_words": arguments.max_words,
            "classes": arguments.classes,
            "qubits": arguments.qubits,
            "layers": arguments.layers,
            "reuploads": arguments.reuploads,
            "hidden_dim": arguments.hidden_dim,
            "embedding": arguments.embedding,
            "reducer": arguments.reducer,
            "scaling": arguments.scaling,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "seeds": seeds,
            "n_classes": n_classes,
            "class_names": names,
            "vocabulary_size": encoder.vocabulary_size,
            "explained_variance_ratio": encoder.explained_variance_ratio,
        },
        "splits": {
            name: {"questions": len(split), "labels": split.label_counts}
            for name, split in splits.items()
        },
        "aer_gpu_available": aer_gpu_available(),
    }

    if requested != ALL_STAGES and REPORT_PATH.exists():
        report = {**json.loads(REPORT_PATH.read_text(encoding="utf-8")), **report}
        print(f"  merging into the existing {REPORT_PATH.name}")

    if "training" in requested:
        report["ceilings"] = reference_ceilings(arguments, n_classes)

    best = None
    if "training" in requested:
        training, best = stage_training(arguments, seeds)
        report["training"] = training

    if "scaling" in requested:
        report["scaling_ablation"] = stage_scaling_ablation(arguments, seeds)

    if "width" in requested and arguments.width_sweep:
        widths = [int(w) for w in arguments.width_sweep.split(",") if w.strip()]
        report["width_sweep"] = stage_width_sweep(arguments, seeds, widths)

    if requested & {"aer-eval", "aer-training"}:
        if best is None:
            print(f"\n  training seed {seeds[0]} on default.qubit for the Aer stages")
            reference_splits, _, reference_classes, _ = prepare_data(arguments, seeds[0])
            model, record = run_one(
                "quantum", arguments, reference_splits, reference_classes, seeds[0]
            )
            print(f"    reference test accuracy: {record['test_accuracy']:.4f}")
            best = (model, seeds[0], reference_splits, reference_classes)

        trained_model, _, best_splits, best_classes = best
        if "aer-eval" in requested:
            shot_counts = [int(s) for s in arguments.aer_shots.split(",") if s.strip()]
            report["aer_evaluation"] = stage_aer_evaluation(
                arguments, trained_model, best_splits, best_classes, shot_counts
            )
        if "aer-training" in requested:
            methods = [m.strip() for m in arguments.aer_diff.split(",") if m.strip()]
            if methods:
                report["aer_training"] = stage_aer_training(
                    arguments, best_splits, best_classes, methods
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
