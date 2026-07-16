#!/usr/bin/env python3
"""Generate a deterministic synthetic FASTA for the public example."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated" / "example_genome.fa"


def generate_reference(path: Path, length: int = 400_000, seed: int = 42) -> None:
    if length < 320_000:
        raise ValueError("Example reference length must be at least 320,000 bases")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(">chrDemo synthetic_reference_for_workflow_testing\n")
        remaining = length
        while remaining:
            line_length = min(80, remaining)
            handle.write("".join(rng.choices("ACGT", k=line_length)) + "\n")
            remaining -= line_length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--length", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_reference(args.output, length=args.length, seed=args.seed)
    print(f"Synthetic FASTA written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
