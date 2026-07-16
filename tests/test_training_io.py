from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attentiontad.training import load_dataset, write_history  # noqa: E402


class TrainingIOTests(unittest.TestCase):
    def test_load_dataset_accepts_preprocessor_output_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.npz"
            np.savez_compressed(
                dataset,
                X_train=np.zeros((4, 100, 4), dtype=np.uint8),
                X_test=np.zeros((2, 100, 4), dtype=np.uint8),
                y_train=np.asarray([0, 1, 0, 1], dtype=np.uint8),
                y_test=np.asarray([0, 1], dtype=np.uint8),
            )
            X_train, X_test, y_train, y_test = load_dataset(dataset)

        self.assertEqual(X_train.shape, (4, 100, 4))
        self.assertEqual(X_test.shape, (2, 100, 4))
        self.assertEqual(y_train.tolist(), [0, 1, 0, 1])
        self.assertEqual(y_test.tolist(), [0, 1])

    def test_load_dataset_rejects_missing_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.npz"
            np.savez_compressed(dataset, X_train=np.zeros((1, 10, 4)))
            with self.assertRaisesRegex(ValueError, "missing required arrays"):
                load_dataset(dataset)

    def test_write_history_uses_one_based_epoch_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "history.csv"
            write_history({"loss": [0.8, 0.4]}, output)
            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(rows[0], "epoch,loss")
        self.assertEqual(rows[1], "1,0.8")
        self.assertEqual(rows[2], "2,0.4")


if __name__ == "__main__":
    unittest.main()
