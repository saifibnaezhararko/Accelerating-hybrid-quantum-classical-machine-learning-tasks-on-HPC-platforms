from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from lambeq import Dataset, PytorchTrainer
from sklearn.metrics import classification_report

from qnlp_hpc.mc1_spider.config import (
    BATCH_SIZE,
    DATA_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    CLASSIFIER_HIDDEN_DIM,
    CLASSIFIER_DROPOUT,
    NOUN_DIM,
    SENTENCE_DIM,
    MAX_ORDER,
    LOG_DIR,
    OUTPUT_DIR,
    SEED,
    WEIGHT_DECAY,
)

from qnlp_hpc.mc1_spider.data import (
    load_mc1,
    set_random_seeds,
    split_data,
)

from qnlp_hpc.mc1_spider.diagrams import (
    build_tensor_diagrams,
    make_pairs_and_targets,
)

from qnlp_hpc.mc1_spider.model import (
    LambeqCrossEntropyLoss,
    SpiderPairModel,
    accuracy,
    evaluation_loss,
)

from qnlp_hpc.mc1_spider.reporting import (
    get_best_epoch,
    save_four_benchmark_plots,
    to_float_list,
)

def run_experiment() -> dict[str, object]:
    set_random_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device_id = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"

    print(f"PyTorch: {torch.__version__}")
    print(f"Training device: {device_name}")
    print(f"Reading: {DATA_PATH}")
    print(
        "Training configuration: "
        f"lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"hidden_dim={CLASSIFIER_HIDDEN_DIM}, "
        f"dropout={CLASSIFIER_DROPOUT}, "
        f"patience={EARLY_STOPPING_PATIENCE}"
    )

    frame = load_mc1(DATA_PATH)
    print(f"Loaded {len(frame)} examples.")

    train_frame, development_frame, test_frame = split_data(frame)
    for name, split in (
        ("training", train_frame),
        ("development", development_frame),
        ("test", test_frame),
    ):
        counts = split["label"].value_counts().sort_index().to_dict()
        print(f"{name:>11}: {len(split):3d} examples | labels {counts}")

    tensor_diagrams = build_tensor_diagrams(frame)

    train_pairs, train_targets = make_pairs_and_targets(
        train_frame, tensor_diagrams
    )
    development_pairs, development_targets = make_pairs_and_targets(
        development_frame, tensor_diagrams
    )
    test_pairs, test_targets = make_pairs_and_targets(
        test_frame, tensor_diagrams
    )

    # Include all diagrams so every symbol needed at evaluation time exists in
    # the model. No development/test labels are used to optimise the model.
    all_tensor_diagrams = [
        diagram
        for pair in train_pairs + development_pairs + test_pairs
        for diagram in pair
    ]

    model = SpiderPairModel.from_diagrams(all_tensor_diagrams)
    model.initialise_weights()
    print(f"Trainable lambeq symbols: {len(model.symbols)}")

    trainer = PytorchTrainer(
        model=model,
        loss_function=LambeqCrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        optimizer_args={"weight_decay": WEIGHT_DECAY},
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device_id,
        evaluate_functions={
            "acc": accuracy,
            "loss": evaluation_loss,
        },
        evaluate_on_train=True,
        log_dir=LOG_DIR,
        verbose="text",
        seed=SEED,
    )

    train_dataset = Dataset(
        train_pairs,
        train_targets,
        batch_size=BATCH_SIZE,
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
        # "loss" is valid because it is registered above in
        # evaluate_functions. Lower cross-entropy is better.
        early_stopping_criterion="loss",
        early_stopping_interval=EARLY_STOPPING_PATIENCE,
        minimize_criterion=True,
    )
    training_seconds = time.perf_counter() - training_start
    print(f"Training completed in {training_seconds:.2f} s.")

    train_loss = to_float_list(trainer.train_epoch_costs)

    # Use the registered evaluation metric so that plotted/selected loss is
    # exactly the same criterion used by lambeq early stopping.
    development_loss = to_float_list(
        trainer.val_eval_results["loss"]
    )
    train_accuracy = to_float_list(trainer.train_eval_results["acc"])
    development_accuracy = to_float_list(
        trainer.val_eval_results["acc"]
    )

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

    history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(train_loss) + 1),
            "train_loss": train_loss,
            "development_loss": development_loss,
            "train_accuracy": train_accuracy,
            "development_accuracy": development_accuracy,
        }
    )
    best_epoch = get_best_epoch(history)
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]

    history.to_csv(
        OUTPUT_DIR / "mc1_spider_benchmark_history.csv",
        index=False,
    )
    save_four_benchmark_plots(history, best_epoch)

    # lambeq's early stopping stores the best validation checkpoint here.
    trainer_best_model = Path(trainer.log_dir) / "best_model.lt"
    if trainer_best_model.is_file():
        model.load(str(trainer_best_model))
        print(f"Loaded best checkpoint: {trainer_best_model}")
    else:
        print(
            "Warning: best_model.lt was not found; evaluating the final "
            "epoch model instead."
        )

    model.eval()
    with torch.no_grad():
        test_logits = model(test_pairs)
        test_probabilities = torch.softmax(test_logits, dim=1)
        test_predictions = (
            torch.argmax(test_probabilities, dim=1).detach().cpu().numpy()
        )

    test_true = test_targets.astype(np.int64, copy=False)
    test_accuracy_value = float(np.mean(test_predictions == test_true))

    print(f"Best epoch: {best_epoch}")
    print(
        "Best development metrics: "
        f"accuracy={best_row['development_accuracy']:.4f}, "
        f"loss={best_row['development_loss']:.4f}"
    )
    print(f"Test accuracy: {test_accuracy_value:.4f}")
    print(
        classification_report(
            test_true,
            test_predictions,
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
    results["predicted_label"] = test_predictions
    results["correct"] = results["label"] == results["predicted_label"]
    results["probability_label_0"] = (
        test_probabilities[:, 0].detach().cpu().numpy()
    )
    results["probability_label_1"] = (
        test_probabilities[:, 1].detach().cpu().numpy()
    )
    results.to_csv(
        OUTPUT_DIR / "mc1_spider_test_predictions.csv",
        index=False,
    )

    final_lambeq_model_path = OUTPUT_DIR / "mc1_spider_best_model.lt"
    model.save(str(final_lambeq_model_path))
    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "mc1_spider_pair_model.pt",
    )

    summary = pd.DataFrame(
        [
            {
                "seed": SEED,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_train_loss": float(best_row["train_loss"]),
                "best_development_loss": float(
                    best_row["development_loss"]
                ),
                "best_train_accuracy": float(best_row["train_accuracy"]),
                "best_development_accuracy": float(
                    best_row["development_accuracy"]
                ),
                "test_accuracy": test_accuracy_value,
                "training_seconds": training_seconds,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "noun_dim": NOUN_DIM,
                "sentence_dim": SENTENCE_DIM,
                "classifier_hidden_dim": CLASSIFIER_HIDDEN_DIM,
                "classifier_dropout": CLASSIFIER_DROPOUT,
                "early_stopping_criterion": "loss",
                "early_stopping_minimize": True,
                "best_epoch_selection": "minimum_development_loss",
                "optimizer": "Adam",
            }
        ]
    )
    summary.to_csv(
        OUTPUT_DIR / "mc1_spider_training_summary.csv",
        index=False,
    )

    return {
        "model": model,
        "trainer": trainer,
        "history": history,
        "test_predictions": results,
        "summary": summary,
        "output_dir": OUTPUT_DIR,
        "test_accuracy": test_accuracy_value,
    }
