"""Run the TREC MiniLM -> PCA -> angle-encoded VQC experiment."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from qnlp_hpc.paths import DATA_DIR, OUTPUTS_DIR, resolve
from qnlp_hpc.trec_pca_vqc.experiment import (
    run_classical_baseline,
    run_vqc_experiment,
)
from qnlp_hpc.trec_pca_vqc.prepare_data import (
    ID_TO_LABEL,
    build_splits,
    class_counts,
    prepare_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TREC six-class classification using MiniLM, PCA, "
            "angle encoding, and a variational quantum circuit."
        )
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=DATA_DIR / "train_5500.label",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DATA_DIR / "TREC_10.label",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--development-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--components",
        type=int,
        default=8,
        choices=(8, 12, 16),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory.",
    )

    return parser


def selected_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but torch.cuda.is_available() is False."
        )

    return requested


def create_output_directory(
    explicit_path: Path | None,
    components: int,
    seed: int,
) -> Path:
    if explicit_path is not None:
        output_dir = resolve(explicit_path)
    else:
        output_dir = (
            OUTPUTS_DIR
            / "trec_pca_vqc"
            / f"pca{components}_seed{seed}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_confusion_matrix(
    matrix: np.ndarray,
    destination: Path,
) -> None:
    labels = [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))]

    pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ).to_csv(destination)


def main() -> int:
    args = build_parser().parse_args()
    device = selected_device(args.device)

    output_dir = create_output_directory(
        args.output_dir,
        components=args.components,
        seed=args.seed,
    )

    print(f"Outputs: {output_dir}")
    print(f"Selected device: {device}")

    training, development, test = build_splits(
        training_path=resolve(args.train_data),
        test_path=resolve(args.test_data),
        samples_per_class=args.samples_per_class,
        development_ratio=args.development_ratio,
        seed=args.seed,
    )

    print("\nDataset splits")
    print("--------------")
    print(f"Training: {len(training)} {class_counts(training)}")
    print(f"Development: {len(development)} {class_counts(development)}")
    print(f"Official test: {len(test)} {class_counts(test)}")

    training.to_csv(output_dir / "training_split.csv", index=False)
    development.to_csv(output_dir / "development_split.csv", index=False)
    test.to_csv(output_dir / "official_test_split.csv", index=False)

    features = prepare_features(
        training,
        development,
        test,
        n_components=args.components,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=device,
    )

    np.savez_compressed(
        output_dir / "prepared_features.npz",
        training_embeddings=features["training_embeddings"],
        development_embeddings=features["development_embeddings"],
        test_embeddings=features["test_embeddings"],
        training_angles=features["training_angles"],
        development_angles=features["development_angles"],
        test_angles=features["test_angles"],
        training_labels=training["label"].to_numpy(),
        development_labels=development["label"].to_numpy(),
        test_labels=test["label"].to_numpy(),
    )

    joblib.dump(features["pca"], output_dir / "pca.joblib")
    joblib.dump(
        features["angle_scaler"],
        output_dir / "angle_scaler.joblib",
    )

    pca = features["pca"]
    pd.DataFrame(
        {
            "component": np.arange(
                1,
                len(pca.explained_variance_ratio_) + 1,
            ),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    ).to_csv(output_dir / "pca_summary.csv", index=False)

    classical_result = run_classical_baseline(
        features["training_angles"],
        features["development_angles"],
        features["test_angles"],
        training,
        development,
        test,
        seed=args.seed,
    )

    joblib.dump(
        classical_result["model"],
        output_dir / "logistic_regression.joblib",
    )
    classical_result["test_predictions"].to_csv(
        output_dir / "classical_test_predictions.csv",
        index=False,
    )
    save_confusion_matrix(
        classical_result["test_metrics"]["confusion_matrix"],
        output_dir / "classical_confusion_matrix.csv",
    )

    vqc_result = run_vqc_experiment(
        features["training_angles"],
        features["development_angles"],
        features["test_angles"],
        training,
        development,
        test,
        n_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device_name=device,
    )

    vqc_result["history"].to_csv(
        output_dir / "vqc_training_history.csv",
        index=False,
    )
    vqc_result["test_predictions"].to_csv(
        output_dir / "vqc_test_predictions.csv",
        index=False,
    )
    save_confusion_matrix(
        vqc_result["test_metrics"]["confusion_matrix"],
        output_dir / "vqc_confusion_matrix.csv",
    )

    torch.save(
        {
            "state_dict": vqc_result["model"].state_dict(),
            "n_qubits": vqc_result["n_qubits"],
            "n_layers": vqc_result["n_layers"],
            "n_classes": 6,
            "seed": args.seed,
        },
        output_dir / "vqc_best_model.pt",
    )

    configuration = {
        "train_data": str(resolve(args.train_data)),
        "test_data": str(resolve(args.test_data)),
        "samples_per_class": args.samples_per_class,
        "development_ratio": args.development_ratio,
        "components": args.components,
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "device": device,
        "embedding_model": args.embedding_model,
        "normalize_embeddings": True,
        "angle_range": [-float(np.pi), float(np.pi)],
        "label_mapping": ID_TO_LABEL,
    }

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "scikit_learn": sklearn.__version__,
    }

    summary = {
        "retained_variance": features["retained_variance"],
        "classical_development_accuracy": (
            classical_result["development_metrics"]["accuracy"]
        ),
        "classical_development_macro_f1": (
            classical_result["development_metrics"]["macro_f1"]
        ),
        "classical_test_accuracy": (
            classical_result["test_metrics"]["accuracy"]
        ),
        "classical_test_macro_f1": (
            classical_result["test_metrics"]["macro_f1"]
        ),
        "classical_training_seconds": (
            classical_result["training_seconds"]
        ),
        "vqc_best_epoch": vqc_result["best_epoch"],
        "vqc_development_accuracy": (
            vqc_result["development_metrics"]["accuracy"]
        ),
        "vqc_development_macro_f1": (
            vqc_result["development_metrics"]["macro_f1"]
        ),
        "vqc_test_accuracy": vqc_result["test_metrics"]["accuracy"],
        "vqc_test_macro_f1": vqc_result["test_metrics"]["macro_f1"],
        "vqc_training_seconds": vqc_result["training_seconds"],
    }

    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "experiment_summary.csv",
        index=False,
    )

    print("\nExperiment complete")
    print("-------------------")
    print(
        f"Classical test: accuracy="
        f"{summary['classical_test_accuracy']:.4f}, "
        f"macro_f1={summary['classical_test_macro_f1']:.4f}"
    )
    print(
        f"VQC test: accuracy={summary['vqc_test_accuracy']:.4f}, "
        f"macro_f1={summary['vqc_test_macro_f1']:.4f}"
    )
    print(f"Results saved in: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
