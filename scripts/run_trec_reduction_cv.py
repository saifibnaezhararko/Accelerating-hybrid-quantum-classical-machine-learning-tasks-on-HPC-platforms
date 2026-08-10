"""Compare PCA, LDA, and NCA using leakage-safe stratified CV."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold

from qnlp_hpc.paths import DATA_DIR, OUTPUTS_DIR, resolve
from qnlp_hpc.trec_pca_vqc.cv_experiment import (
    run_fixed_epoch_vqc,
    run_fold_logistic_regression,
)
from qnlp_hpc.trec_pca_vqc.prepare_data import (
    encode_texts,
    load_trec,
    stratified_sample,
)
from qnlp_hpc.trec_pca_vqc.reduction import (
    SUPPORTED_METHODS,
    fit_transform_fold,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe comparison of PCA-5, LDA-5, and NCA-5 "
            "using identical cross-validation folds and VQC settings."
        )
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=DATA_DIR / "train_5500.label",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(SUPPORTED_METHODS),
    )
    parser.add_argument(
        "--components",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
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
    )
    parser.add_argument(
        "--measurement-mode",
        choices=("z", "z_zz"),
        default="z",
    )
    parser.add_argument(
        "--data-reuploading",
        action="store_true",
        help="Re-encode input angles before each additional VQC layer.",
    )
    parser.add_argument(
        "--use-full-training",
        action="store_true",
        help="Use all TREC training examples instead of balanced sampling.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        help="Reuse cached embeddings instead of loading SentenceTransformer.",
    )
    return parser


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return requested


def make_output_directory(
    explicit_path: Path | None,
    seed: int,
) -> Path:
    if explicit_path is not None:
        output_dir = resolve(explicit_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            OUTPUTS_DIR
            / "trec_reduction_cv"
            / f"{timestamp}_seed{seed}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    class_names = ("ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM")

    pd.DataFrame(
        matrix,
        index=[f"actual_{name}" for name in class_names],
        columns=[f"predicted_{name}" for name in class_names],
    ).to_csv(output_path)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and sample standard deviation across folds."""
    rows: list[dict[str, object]] = []

    for (method, model), group in results.groupby(
        ["method", "model"],
        sort=False,
    ):
        rows.append(
            {
                "method": method,
                "model": model,
                "folds": len(group),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "training_seconds_mean": (
                    group["training_seconds"].mean()
                ),
                "training_seconds_std": (
                    group["training_seconds"].std(ddof=1)
                ),
                "pca_explained_variance_mean": (
                    group["explained_variance"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = build_parser().parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be at least 2.")

    if args.components != 5 and "lda" in args.methods:
        raise SystemExit(
            "The six-class LDA comparison must use 5 components."
        )

    device = select_device(args.device)
    output_dir = make_output_directory(
        args.output_dir,
        seed=args.seed,
    )

    vqc_model_name = f"vqc_{args.measurement_mode}"

    if args.data_reuploading:
        vqc_model_name += "_reupload"

    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Methods: {args.methods}")
    print(f"VQC model: {vqc_model_name}")

    full_training = load_trec(resolve(args.train_data))
    if args.use_full_training:
        sampled = full_training.sample(
            frac=1.0,
            random_state=args.seed,
        ).reset_index(drop=True)
    else:
        sampled = stratified_sample(
            full_training,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )

    sampled.to_csv(
        output_dir / "sampled_cv_dataset.csv",
        index=False,
    )

    print(f"Sampled examples: {len(sampled)}")
    print(
        "Class counts:",
        sampled["coarse_label"].value_counts().sort_index().to_dict(),
    )
    print(
        "Important: the official TREC test set is not loaded or used "
        "during this CV experiment."
    )

    labels = sampled["label"].to_numpy(dtype=np.int64)

    if args.embeddings_npz is not None:
        embeddings_path = resolve(args.embeddings_npz)
        print(f"Loading cached embeddings: {embeddings_path}")

        cached = np.load(embeddings_path)
        embeddings = np.asarray(
            cached["embeddings"],
            dtype=np.float32,
        )
        cached_labels = np.asarray(
            cached["labels"],
            dtype=np.int64,
        )

        if len(embeddings) != len(sampled):
            raise SystemExit(
                "Cached embeddings do not match the sampled dataset size: "
                f"{len(embeddings)} != {len(sampled)}"
            )

        if not np.array_equal(cached_labels, labels):
            raise SystemExit(
                "Cached embedding labels do not match the current dataset order. "
                "Use the same --seed and data-selection settings."
            )
    else:
        print(f"Loading embedding model: {args.embedding_model}")

        encoder = SentenceTransformer(
            args.embedding_model,
            device=device,
        )

        embeddings = encode_texts(
            sampled,
            encoder,
            batch_size=args.batch_size,
        )

    np.savez_compressed(
        output_dir / "cv_embeddings.npz",
        embeddings=embeddings,
        labels=labels,
    )

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(splitter.split(embeddings, labels))

    result_rows: list[dict[str, object]] = []

    for method in args.methods:
        method_directory = output_dir / method
        method_directory.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 72)
        print(f"Reduction method: {method.upper()}")
        print("=" * 72)

        for fold_number, (
            training_indices,
            validation_indices,
        ) in enumerate(splits, start=1):
            fold_seed = args.seed + fold_number

            print()
            print(
                f"{method.upper()} fold "
                f"{fold_number}/{args.folds}"
            )

            reduction = fit_transform_fold(
                method=method,
                training_embeddings=embeddings[training_indices],
                training_labels=labels[training_indices],
                validation_embeddings=embeddings[validation_indices],
                n_components=args.components,
                seed=fold_seed,
            )

            logistic_result = run_fold_logistic_regression(
                training_features=reduction.training_angles,
                training_labels=labels[training_indices],
                validation_features=reduction.validation_angles,
                validation_labels=labels[validation_indices],
                seed=fold_seed,
            )

            print(
                "  Logistic Regression: "
                f"accuracy={logistic_result['accuracy']:.4f}, "
                f"macro_f1={logistic_result['macro_f1']:.4f}"
            )

            result_rows.append(
                {
                    "method": method,
                    "model": "logistic_regression",
                    "fold": fold_number,
                    "seed": fold_seed,
                    "training_examples": len(training_indices),
                    "validation_examples": len(validation_indices),
                    "accuracy": logistic_result["accuracy"],
                    "macro_f1": logistic_result["macro_f1"],
                    "training_seconds": (
                        logistic_result["training_seconds"]
                    ),
                    "explained_variance": (
                        reduction.explained_variance
                    ),
                }
            )

            save_confusion_matrix(
                logistic_result["confusion_matrix"],
                method_directory
                / f"fold_{fold_number}_logistic_confusion.csv",
            )

            print("  Training fixed-epoch VQC...")

            vqc_result = run_fixed_epoch_vqc(
                training_features=reduction.training_angles,
                training_labels=labels[training_indices],
                validation_features=reduction.validation_angles,
                validation_labels=labels[validation_indices],
                n_layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=fold_seed,
                device_name=device,
                measurement_mode=args.measurement_mode,
                data_reuploading=args.data_reuploading,
            )

            print(
                "  VQC validation: "
                f"accuracy={vqc_result['accuracy']:.4f}, "
                f"macro_f1={vqc_result['macro_f1']:.4f}"
            )

            result_rows.append(
                {
                    "method": method,
                    "model": vqc_model_name,
                    "fold": fold_number,
                    "seed": fold_seed,
                    "training_examples": len(training_indices),
                    "validation_examples": len(validation_indices),
                    "accuracy": vqc_result["accuracy"],
                    "macro_f1": vqc_result["macro_f1"],
                    "training_seconds": (
                        vqc_result["training_seconds"]
                    ),
                    "explained_variance": (
                        reduction.explained_variance
                    ),
                }
            )

            save_confusion_matrix(
                vqc_result["confusion_matrix"],
                method_directory
                / f"fold_{fold_number}_{vqc_model_name}_confusion.csv",
            )


    results = pd.DataFrame(result_rows)
    summary = build_summary(results)

    results.to_csv(
        output_dir / "fold_results.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
    )

    configuration = {
        "train_data": str(resolve(args.train_data)),
        "official_test_used": False,
        "samples_per_class": args.samples_per_class,
        "methods": args.methods,
        "components": args.components,
        "folds": args.folds,
        "layers": args.layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device,
        "embedding_model": args.embedding_model,
        "embedding_fitted_in_cv": False,
        "reduction_fitted_inside_each_fold": True,
        "angle_scaler_fitted_inside_each_fold": True,
        "measurement_mode": args.measurement_mode,
        "data_reuploading": args.data_reuploading,
        "use_full_training": args.use_full_training,
        "class_weighting": "balanced",
    }

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "scikit_learn": sklearn.__version__,
    }

    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    print()
    print("Cross-validation summary")
    print("------------------------")
    print(summary.to_string(index=False))
    print(f"Results saved in: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())