"""Prepare fixed-length sequence windows for AttentionTAD."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TextIO

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A zero-based, half-open genomic interval."""

    chrom: str
    start: int
    end: int
    name: str = "."

    @property
    def length(self) -> int:
        return self.end - self.start


def _open_text(path: str | Path) -> TextIO:
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.open(source, "rt", encoding="utf-8")
    return source.open("r", encoding="utf-8")


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into an in-memory chromosome dictionary."""

    genome: dict[str, str] = {}
    current_name: str | None = None
    chunks: list[str] = []

    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    genome[current_name] = "".join(chunks).upper()
                current_name = line[1:].split()[0]
                if not current_name:
                    raise ValueError(f"Missing FASTA record name at line {line_number}")
                if current_name in genome:
                    raise ValueError(f"Duplicate FASTA record name: {current_name}")
                chunks = []
            elif current_name is None:
                raise ValueError("FASTA sequence encountered before the first header")
            else:
                chunks.append(line)

    if current_name is not None:
        genome[current_name] = "".join(chunks).upper()
    if not genome:
        raise ValueError(f"No FASTA records found in {path}")
    return genome


def parse_bed(path: str | Path) -> list[Interval]:
    """Parse the first four columns of a BED-like text file."""

    intervals: list[Interval] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"BED line {line_number} has fewer than three columns")
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"BED line {line_number} contains non-integer coordinates"
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(f"BED line {line_number} contains an invalid interval")
            name = fields[3] if len(fields) >= 4 else f"interval_{line_number}"
            intervals.append(Interval(fields[0], start, end, name))

    if not intervals:
        raise ValueError(f"No intervals found in {path}")
    return intervals


def normalize_interval(interval: Interval, window_size: int) -> Interval:
    """Center an interval in a fixed-size window."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if interval.length == window_size:
        return interval
    midpoint = (interval.start + interval.end) // 2
    start = max(0, midpoint - window_size // 2)
    return Interval(interval.chrom, start, start + window_size, interval.name)


def valid_positive_windows(
    intervals: Iterable[Interval],
    genome: Mapping[str, str],
    window_size: int,
) -> list[Interval]:
    """Normalize intervals and retain windows represented in the FASTA file."""

    valid: list[Interval] = []
    for interval in intervals:
        if interval.chrom not in genome:
            continue
        chrom_length = len(genome[interval.chrom])
        if chrom_length < window_size:
            continue
        normalized = normalize_interval(interval, window_size)
        if normalized.end > chrom_length:
            normalized = Interval(
                normalized.chrom,
                chrom_length - window_size,
                chrom_length,
                normalized.name,
            )
        valid.append(normalized)

    if not valid:
        raise ValueError(
            "No BED intervals could be matched to a full-length window in the FASTA file"
        )
    return valid


def deduplicate_windows(windows: Sequence[Interval]) -> list[Interval]:
    """Keep the first interval for each unique genomic window."""

    unique: list[Interval] = []
    seen: set[tuple[str, int, int]] = set()
    for interval in windows:
        key = (interval.chrom, interval.start, interval.end)
        if key not in seen:
            seen.add(key)
            unique.append(interval)
    return unique


def validate_non_overlapping(windows: Sequence[Interval], label: str) -> None:
    """Reject overlapping windows so sequence cannot cross dataset partitions."""

    if not windows:
        return
    by_chrom: dict[str, list[Interval]] = {}
    for interval in windows:
        by_chrom.setdefault(interval.chrom, []).append(interval)

    for chrom, chrom_windows in by_chrom.items():
        ordered = sorted(chrom_windows, key=lambda item: (item.start, item.end))
        previous = ordered[0]
        for current in ordered[1:]:
            if current.start < previous.end:
                raise ValueError(
                    f"Overlapping {label} windows after normalization: "
                    f"{chrom}:{previous.start}-{previous.end} and "
                    f"{chrom}:{current.start}-{current.end}"
                )
            if current.end > previous.end:
                previous = current


def _overlaps(interval: Interval, candidates: Sequence[Interval]) -> bool:
    return any(
        interval.start < candidate.end and candidate.start < interval.end
        for candidate in candidates
    )


def generate_negative_windows(
    positive_windows: Sequence[Interval],
    genome: Mapping[str, str],
    window_size: int,
    seed: int = 42,
    count: int | None = None,
) -> list[Interval]:
    """Generate deterministic windows that do not overlap positive intervals."""

    target_count = len(positive_windows) if count is None else count
    if target_count <= 0:
        raise ValueError("count must be positive")

    positives_by_chrom: dict[str, list[Interval]] = {}
    for interval in positive_windows:
        positives_by_chrom.setdefault(interval.chrom, []).append(interval)

    eligible_chroms = [
        chrom for chrom, sequence in genome.items() if len(sequence) >= window_size
    ]
    if not eligible_chroms:
        raise ValueError("No FASTA sequence is long enough for the requested window size")

    available_starts = np.asarray(
        [len(genome[chrom]) - window_size + 1 for chrom in eligible_chroms],
        dtype=np.float64,
    )
    probabilities = available_starts / available_starts.sum()
    rng = np.random.default_rng(seed)
    negatives: list[Interval] = []
    negatives_by_chrom: dict[str, list[Interval]] = {}
    seen: set[tuple[str, int]] = set()
    max_attempts = max(1_000, target_count * 200)

    for attempt in range(max_attempts):
        if len(negatives) >= target_count:
            break
        chrom = str(rng.choice(eligible_chroms, p=probabilities))
        max_start = len(genome[chrom]) - window_size
        start = int(rng.integers(0, max_start + 1))
        key = (chrom, start)
        if key in seen:
            continue
        seen.add(key)
        candidate = Interval(
            chrom,
            start,
            start + window_size,
            f"synthetic_negative_{len(negatives) + 1}",
        )
        if _overlaps(candidate, positives_by_chrom.get(chrom, [])):
            continue
        if _overlaps(candidate, negatives_by_chrom.get(chrom, [])):
            continue
        negatives.append(candidate)
        negatives_by_chrom.setdefault(chrom, []).append(candidate)

    if len(negatives) != target_count:
        raise RuntimeError(
            f"Generated {len(negatives)} of {target_count} requested negative windows; "
            "use a larger reference region or fewer positive windows"
        )
    return negatives


def one_hot_encode(sequence: str) -> np.ndarray:
    """Encode DNA bases as A/C/G/T channels using compact unsigned bytes."""

    lookup = np.zeros((256, 4), dtype=np.uint8)
    for base, channel in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
        lookup[ord(base), channel] = 1
        lookup[ord(base.lower()), channel] = 1
    encoded_bytes = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    return lookup[encoded_bytes]


def encode_windows(
    windows: Sequence[Interval], genome: Mapping[str, str], window_size: int
) -> np.ndarray:
    """Extract and one-hot encode fixed-length windows."""

    encoded = np.empty((len(windows), window_size, 4), dtype=np.uint8)
    for index, interval in enumerate(windows):
        sequence = genome[interval.chrom][interval.start : interval.end]
        if len(sequence) != window_size:
            raise ValueError(f"Unexpected sequence length for {interval}")
        encoded[index] = one_hot_encode(sequence)
    return encoded


def _stratified_indices(
    labels: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("Each label needs at least two samples for a stratified split")
        indices = rng.permutation(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_count = min(test_count, len(indices) - 1)
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])

    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    return train_indices, test_indices


def write_bed(intervals: Sequence[Interval], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for interval in intervals:
            handle.write(
                f"{interval.chrom}\t{interval.start}\t{interval.end}\t{interval.name}\n"
            )


def preprocess_dataset(
    bed_path: str | Path,
    fasta_path: str | Path,
    output_path: str | Path,
    negative_bed_path: str | Path | None = None,
    window_size: int = 10_000,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    """Create a balanced, compressed AttentionTAD training dataset."""

    genome = read_fasta(fasta_path)
    positives = deduplicate_windows(
        valid_positive_windows(parse_bed(bed_path), genome, window_size)
    )
    validate_non_overlapping(positives, "positive")
    negatives = generate_negative_windows(positives, genome, window_size, seed=seed)
    validate_non_overlapping(negatives, "negative")

    windows = list(positives) + list(negatives)
    labels = np.concatenate(
        [np.ones(len(positives), dtype=np.uint8), np.zeros(len(negatives), dtype=np.uint8)]
    )
    features = encode_windows(windows, genome, window_size)
    train_indices, test_indices = _stratified_indices(labels, test_size, seed)

    metadata = {
        "bed": Path(bed_path).name,
        "fasta": Path(fasta_path).name,
        "window_size": window_size,
        "seed": seed,
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "split_strategy": "stratified_non_overlapping_windows",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X_train=features[train_indices],
        X_test=features[test_indices],
        y_train=labels[train_indices],
        y_test=labels[test_indices],
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    if negative_bed_path is not None:
        write_bed(negatives, negative_bed_path)

    return {
        **metadata,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "output": str(destination),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare BED and FASTA inputs for AttentionTAD training"
    )
    parser.add_argument("--bed", required=True, help="Boundary intervals in BED format")
    parser.add_argument("--fasta", required=True, help="Reference genome in FASTA format")
    parser.add_argument("--output", required=True, help="Compressed output NPZ path")
    parser.add_argument(
        "--negative-bed",
        help="Optional path for the generated negative-window BED file",
    )
    parser.add_argument("--window-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = preprocess_dataset(
        bed_path=args.bed,
        fasta_path=args.fasta,
        output_path=args.output,
        negative_bed_path=args.negative_bed,
        window_size=args.window_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
