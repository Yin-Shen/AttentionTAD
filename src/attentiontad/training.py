"""Train and evaluate the public AttentionTAD model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model import create_attentiontad_model


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "TensorFlow is required for training. "
            "Install the training dependencies with: pip install -e '.[train]'"
        ) from exc
    return tf


def configure_runtime(seed: int, force_cpu: bool = False):
    """Configure deterministic seeds and conservative GPU memory allocation."""

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf = _load_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if not force_cpu:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    return tf


def load_dataset(path: str | Path) -> tuple[np.ndarray, ...]:
    """Load and validate a dataset produced by the preprocessing command."""

    with np.load(path, allow_pickle=False) as data:
        required = ("X_train", "X_test", "y_train", "y_test")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {', '.join(missing)}")
        arrays = tuple(np.asarray(data[key]) for key in required)

    X_train, X_test, y_train, y_test = arrays
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Sequence arrays must have shape (samples, positions, channels)")
    if X_train.shape[-1] != 4 or X_test.shape[-1] != 4:
        raise ValueError("Sequence arrays must use four A/C/G/T channels")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match")
    return X_train, X_test, y_train, y_test


def write_history(history: Mapping[str, Sequence[float]], path: str | Path) -> None:
    """Write Keras history values without requiring pandas."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history)
    epochs = max((len(history[key]) for key in keys), default=0)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for epoch in range(epochs):
            writer.writerow(
                [epoch + 1, *[history[key][epoch] for key in keys]]
            )


def plot_history(history: Mapping[str, Sequence[float]], path: str | Path) -> bool:
    """Create a compact training plot when matplotlib is installed."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return False

    metric_pairs = [
        ("loss", "val_loss", "Loss"),
        ("roc_auc", "val_roc_auc", "ROC AUC"),
        ("pr_auc", "val_pr_auc", "PR AUC"),
    ]
    available = [pair for pair in metric_pairs if pair[0] in history]
    if not available:
        return False

    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for axis, (train_key, validation_key, label) in zip(axes, available):
        axis.plot(history[train_key], label="Training", linewidth=2)
        if validation_key in history:
            axis.plot(history[validation_key], label="Validation", linewidth=2)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def train(
    data_path: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 128,
    validation_split: float = 0.1,
    learning_rate: float = 1e-4,
    seed: int = 42,
    force_cpu: bool = False,
    num_filters: int = 256,
    kernel_size: int = 19,
    dense_units: int = 64,
) -> dict[str, float]:
    """Train the model, save the best checkpoint, and evaluate the test split."""

    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    tf = configure_runtime(seed=seed, force_cpu=force_cpu)
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "attentiontad_model.keras"

    model = create_attentiontad_model(
        input_shape=X_train.shape[1:],
        num_filters=num_filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    monitor = "val_loss" if validation_split > 0 else "loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            mode="min",
            verbose=1,
        )
    ]
    if validation_split > 0:
        callbacks.extend(
            [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7
                ),
            ]
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    write_history(history.history, destination / "training_history.csv")
    plot_history(history.history, destination / "training_history.png")

    best_model = tf.keras.models.load_model(model_path)
    evaluation = best_model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    metrics = {key: float(value) for key, value in evaluation.items()}
    with (destination / "test_metrics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AttentionTAD classifier")
    parser.add_argument("--data", required=True, help="Preprocessed NPZ dataset")
    parser.add_argument("--output-dir", default="outputs/attentiontad")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU use")
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=19)
    parser.add_argument("--dense-units", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        seed=args.seed,
        force_cpu=args.cpu,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        dense_units=args.dense_units,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
