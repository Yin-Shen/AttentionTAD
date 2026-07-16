from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attentiontad.preprocessing import (  # noqa: E402
    Interval,
    generate_negative_windows,
    one_hot_encode,
    parse_bed,
    preprocess_dataset,
    validate_non_overlapping,
)


class PreprocessingTests(unittest.TestCase):
    def test_one_hot_encode_uses_compact_acgt_channels(self):
        encoded = one_hot_encode("ACGTNacgt")
        self.assertEqual(encoded.shape, (9, 4))
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1])
        np.testing.assert_array_equal(encoded[4], [0, 0, 0, 0])
        np.testing.assert_array_equal(encoded[5:], encoded[:4])

    def test_parse_bed_accepts_comments_and_optional_names(self):
        with tempfile.TemporaryDirectory() as directory:
            bed_path = Path(directory) / "example.bed"
            bed_path.write_text(
                "# comment\nchrDemo\t10000\t20000\tboundary_a\n"
                "chrDemo\t30000\t40000\n",
                encoding="utf-8",
            )
            intervals = parse_bed(bed_path)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], Interval("chrDemo", 10000, 20000, "boundary_a"))
        self.assertTrue(intervals[1].name.startswith("interval_"))

    def test_negative_generation_is_deterministic_and_non_overlapping(self):
        genome = {"chrDemo": "A" * 100_000}
        positives = [
            Interval("chrDemo", 10_000, 20_000, "p1"),
            Interval("chrDemo", 40_000, 50_000, "p2"),
        ]
        first = generate_negative_windows(positives, genome, 10_000, seed=7)
        second = generate_negative_windows(positives, genome, 10_000, seed=7)
        self.assertEqual(first, second)
        for negative in first:
            for positive in positives:
                self.assertFalse(
                    negative.start < positive.end and positive.start < negative.end
                )
        for index, negative in enumerate(first):
            for other in first[index + 1 :]:
                self.assertFalse(
                    negative.start < other.end and other.start < negative.end
                )

    def test_overlap_validation_rejects_shared_sequence(self):
        windows = [
            Interval("chrDemo", 10_000, 20_000, "p1"),
            Interval("chrDemo", 15_000, 25_000, "p2"),
        ]
        with self.assertRaisesRegex(ValueError, "Overlapping positive windows"):
            validate_non_overlapping(windows, "positive")

    def test_preprocess_dataset_writes_balanced_compressed_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta_path = root / "reference.fa"
            bed_path = root / "boundaries.bed"
            output_path = root / "dataset.npz"
            negative_path = root / "negative.bed"
            fasta_path.write_text(
                ">chrDemo\n" + "ACGT" * 25_000 + "\n", encoding="utf-8"
            )
            bed_path.write_text(
                "chrDemo\t10000\t20000\tp1\n"
                "chrDemo\t40000\t50000\tp2\n",
                encoding="utf-8",
            )

            summary = preprocess_dataset(
                bed_path,
                fasta_path,
                output_path,
                negative_bed_path=negative_path,
                test_size=0.5,
                seed=11,
            )
            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["X_train"].shape, (2, 10_000, 4))
                self.assertEqual(data["X_test"].shape, (2, 10_000, 4))
                self.assertEqual(data["X_train"].dtype, np.uint8)
                self.assertEqual(int(data["y_train"].sum()), 1)
                self.assertEqual(int(data["y_test"].sum()), 1)

            self.assertTrue(negative_path.exists())
            self.assertEqual(summary["positive_samples"], 2)
            self.assertEqual(summary["negative_samples"], 2)


if __name__ == "__main__":
    unittest.main()
