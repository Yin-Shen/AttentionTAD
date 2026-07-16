# Synthetic example

This directory contains a small synthetic dataset for checking the AttentionTAD preprocessing workflow. It does not contain coordinates or sequences from the study data.

## Files

- `data/example_boundaries.bed`: eight synthetic 10 kb windows on `chrDemo`
- `generate_example_reference.py`: generates a deterministic 400 kb synthetic FASTA
- `generated/`: runtime outputs; ignored by Git

## Run

From the repository root:

```bash
python examples/generate_example_reference.py

python -m attentiontad.preprocessing \
  --bed examples/data/example_boundaries.bed \
  --fasta examples/generated/example_genome.fa \
  --output examples/generated/example_dataset.npz \
  --negative-bed examples/generated/example_negative_windows.bed \
  --seed 42
```

The command prints a short JSON summary and creates a balanced NPZ dataset. Model training is described in the main [README](../README.md).
