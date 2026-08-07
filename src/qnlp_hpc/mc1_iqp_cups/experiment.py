"""Train and evaluate the MC1 IQP sentence-pair model."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lambeq import Dataset, PytorchTrainer
from sklearn.metrics import classification_report

from qnlp_hpc.mc1_iqp_cups import circuits, config, data, model, reporting


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_history(trainer: PytorchTrainer) -> pd.DataFrame:
    """Combine the trainer's loss and accuracy histories."""
    train_loss = reporting.to_float_list(trainer.train_eval_results["loss"])
    development_loss = reporting.to_float_list(trainer.val_eval_results["loss"])
    train_accuracy = reporting.to_float_list(trainer.train_eval_results["acc"])
    development_accuracy = reporting.to_float_list(trainer.val_eval_results["acc"])

    lengths = {
        len(train_loss),
        len(development_loss),
        len(train_accuracy),
        len(development_accuracy),
    }
    if len(lengths) != 1:
        raise RuntimeError(
            "Training histories have inconsistent lengths: "
            f"train_loss={len(train_loss)}, "
            f"development_loss={len(development_loss)}, "
            f"train_accuracy={len(train_accuracy)}, "
            f"development_accuracy={len(development_accuracy)}"
        )

    return pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_loss) + 1),
            "train_loss": train_loss,
            "development_loss": development_loss,
            "train_accuracy": train_accuracy,
            "development_accuracy": development_accuracy,
        }
    )


def evaluate_test_set(
    pair_model: model.IQPPairModel,
    test_pairs: list[tuple[object, object]],
    test_targets: np.ndarray,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Evaluate the selected model and save test predictions."""
    pair_model.eval()
    with torch.no_grad():
        logits = pair_model(test_pairs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1).detach().cpu().numpy()

    expected = test_targets.astype(np.int64, copy=False)
    test_accuracy = float(np.mean(predictions == expected))

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        classification_report(
            expected,
            predictions,
            labels=[0, 1],
            target_names=[
                "different domain (0)",
                "same domain (1)",
            ],
            digits=4,
            zero_division=0,
        )
    )

    results = test_frame.reset_index(drop=True).copy()
    results["predicted_label"] = predictions
    results["correct"] = results["label"] == results["predicted_label"]
    results["probability_label_0"] = probabilities[:, 0].detach().cpu().numpy()
    results["probability_label_1"] = probabilities[:, 1].detach().cpu().numpy()
    results.to_csv(
        config.OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_test_predictions.csv",
        index=False,
    )
    return results, test_accuracy


def run_experiment() -> dict[str, object]:
    """Run training, model selection, testing, and result export."""
    set_random_seeds(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Do not reload a checkpoint produced by an incompatible earlier run.
    old_checkpoint = config.LOG_DIR / "best_model.lt"
    if old_checkpoint.is_file():
        old_checkpoint.unlink()

    print(f"PyTorch: {torch.__version__}")
    print("Training device: CPU (lambeq tensor contraction)")
    print(f"Reading: {config.DATA_PATH}")
    print(
        "Training configuration: "
        f"lr={config.LEARNING_RATE}, "
        f"weight_decay={config.WEIGHT_DECAY}, "
        f"hidden_dim={config.CLASSIFIER_HIDDEN_DIM}, "
        f"dropout={config.CLASSIFIER_DROPOUT}, "
        f"patience={config.EARLY_STOPPING_PATIENCE}, "
        f"iqp_layers={config.IQP_LAYERS}, "
        f"sentence_qubits={config.SENTENCE_QUBITS}"
    )

    frame = data.load_pairs(config.DATA_PATH)
    print(f"Loaded {len(frame)} examples.")

    train_frame, development_frame, test_frame = data.split_stratified(frame)

    print("Created stratified random train/development/test split.")
    print(
        f"Training: {len(train_frame)} | "
        f"Development: {len(development_frame)} | "
        f"Test: {len(test_frame)}"
    )

    data.save_split_diagnostics(
        frame,
        train_frame,
        development_frame,
        test_frame,
        output_dir=config.OUTPUT_DIR,
    )

    for name, split in (
        ("training", train_frame),
        ("development", development_frame),
        ("test", test_frame),
    ):
        label_counts = split["label"].value_counts().sort_index().to_dict()
        print(f"{name:>11}: {len(split):3d} examples | labels {label_counts}")

    retained_frame = pd.concat(
        [train_frame, development_frame, test_frame],
        ignore_index=True,
    )
    circuit_lookup = circuits.build_quantum_circuits(retained_frame)

    train_pairs, train_targets = circuits.make_pairs_and_targets(
        train_frame,
        circuit_lookup,
    )
    development_pairs, development_targets = circuits.make_pairs_and_targets(
        development_frame,
        circuit_lookup,
    )
    test_pairs, test_targets = circuits.make_pairs_and_targets(
        test_frame,
        circuit_lookup,
    )

    training_circuits = [circuit for pair in train_pairs for circuit in pair]
    evaluation_circuits = [circuit for pair in development_pairs + test_pairs for circuit in pair]
    training_symbols = circuits.collect_circuit_symbols(training_circuits)
    evaluation_symbols = circuits.collect_circuit_symbols(evaluation_circuits)

    missing_symbols = evaluation_symbols.difference(training_symbols)

    print(f"Training circuit symbols: {len(training_symbols)}")
    print(f"Development/test circuit symbols: {len(evaluation_symbols)}")
    print(f"Unseen development/test symbols: {len(missing_symbols)}")

    if missing_symbols:
        print("First 20 unseen symbols:")
        for symbol in sorted(map(str, missing_symbols))[:20]:
            print(f"  {symbol}")

        raise RuntimeError(
            f"Found {len(missing_symbols)} symbols in development/test "
            "that never appear in training."
        )

    # Only training circuits determine the trainable symbol table.
    pair_model = model.IQPPairModel.from_diagrams(training_circuits)
    pair_model.initialise_weights()
    print(f"Trainable lambeq symbols: {len(pair_model.symbols)}")
    print(f"Circuit output dimension: {config.CIRCUIT_OUTPUT_DIM}")

    trainer = PytorchTrainer(
        model=pair_model,
        loss_function=model.LambeqCrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_args={"weight_decay": config.WEIGHT_DECAY},
        learning_rate=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        device=-1,
        evaluate_functions={
            "acc": model.accuracy,
            "loss": model.evaluation_loss,
        },
        evaluate_on_train=True,
        log_dir=config.LOG_DIR,
        verbose="text",
        seed=config.SEED,
    )

    train_dataset = Dataset(
        train_pairs,
        train_targets,
        batch_size=min(config.BATCH_SIZE, len(train_pairs)),
        shuffle=True,
    )
    development_dataset = Dataset(
        development_pairs,
        development_targets,
        batch_size=len(development_pairs),
        shuffle=False,
    )

    training_start = time.perf_counter()
    trainer.fit(
        train_dataset,
        development_dataset,
        eval_interval=1,
        log_interval=5,
        early_stopping_criterion="loss",
        early_stopping_interval=config.EARLY_STOPPING_PATIENCE,
        minimize_criterion=True,
    )
    training_seconds = time.perf_counter() - training_start
    print(f"Training completed in {training_seconds:.2f} s.")

    history = build_history(trainer)
    best_epoch = reporting.get_best_epoch(history)
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    history.to_csv(
        config.OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_history.csv",
        index=False,
    )
    reporting.save_four_benchmark_plots(
        history,
        best_epoch,
        output_dir=config.OUTPUT_DIR,
    )

    best_checkpoint = Path(trainer.log_dir) / "best_model.lt"
    if best_checkpoint.is_file():
        pair_model.load(str(best_checkpoint))
        print(f"Loaded best checkpoint: {best_checkpoint}")
    else:
        print("Warning: best_model.lt was not found; evaluating the final " "epoch model instead.")

    train_prediction_counts = model.prediction_counts(
        pair_model,
        train_pairs,
    )
    development_prediction_counts = model.prediction_counts(
        pair_model,
        development_pairs,
    )
    test_prediction_counts = model.prediction_counts(pair_model, test_pairs)
    print(f"Training prediction counts: {train_prediction_counts}")
    print(f"Development prediction counts: {development_prediction_counts}")
    print(f"Test prediction counts: {test_prediction_counts}")

    print(f"Selected epoch: {best_epoch} (minimum development loss)")
    print(
        "Selected development metrics: "
        f"accuracy={best_row['development_accuracy']:.4f}, "
        f"loss={best_row['development_loss']:.4f}"
    )
    test_results, test_accuracy = evaluate_test_set(
        pair_model,
        test_pairs,
        test_targets,
        test_frame,
    )

    pair_model.save(str(config.OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_best_model.lt"))
    torch.save(
        pair_model.state_dict(),
        config.OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_pair_model.pt",
    )

    summary = pd.DataFrame(
        [
            {
                "reader": "cups_reader",
                "split_strategy": "stratified_random",
                "test_ratio": config.TEST_RATIO,
                "development_ratio": config.DEVELOPMENT_RATIO,
                "training_pairs": len(train_frame),
                "development_pairs": len(development_frame),
                "test_pairs": len(test_frame),
                "seed": config.SEED,
                "epochs_completed": len(history),
                "selected_epoch": best_epoch,
                "selected_train_loss": float(best_row["train_loss"]),
                "selected_development_loss": float(best_row["development_loss"]),
                "selected_train_accuracy": float(best_row["train_accuracy"]),
                "selected_development_accuracy": float(best_row["development_accuracy"]),
                "test_accuracy": test_accuracy,
                "training_seconds": training_seconds,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "weight_decay": config.WEIGHT_DECAY,
                "sentence_qubits": config.SENTENCE_QUBITS,
                "iqp_layers": config.IQP_LAYERS,
                "n_single_qubit_params": config.N_SINGLE_QUBIT_PARAMS,
                "circuit_output_dim": config.CIRCUIT_OUTPUT_DIM,
                "quantum_backend": ("PytorchQuantumModel tensor contraction"),
                "classifier_hidden_dim": config.CLASSIFIER_HIDDEN_DIM,
                "classifier_dropout": config.CLASSIFIER_DROPOUT,
                "early_stopping_criterion": "development_loss",
                "training_prediction_counts": str(train_prediction_counts),
                "development_prediction_counts": str(development_prediction_counts),
                "test_prediction_counts": str(test_prediction_counts),
            }
        ]
    )
    summary.to_csv(
        config.OUTPUT_DIR / "mc1_iqp_cups_sentence_disjoint_training_summary.csv",
        index=False,
    )

    print(f"Outputs saved in: {config.OUTPUT_DIR}")
    return {
        "model": pair_model,
        "trainer": trainer,
        "history": history,
        "test_predictions": test_results,
        "summary": summary,
        "output_dir": config.OUTPUT_DIR,
        "test_accuracy": test_accuracy,
    }
