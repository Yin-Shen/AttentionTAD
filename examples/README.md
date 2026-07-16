# AttentionTAD synthetic example

_A small, deterministic fixture for checking the public preprocessing workflow._

---

## 🧬 Contents

- `data/example_boundaries.bed` contains eight synthetic 10 kb boundary windows on `chrDemo`
- `generate_example_reference.py` creates a deterministic 400 kb synthetic FASTA
- `generated/` stores runtime BED, NPZ, model, and metric outputs and is ignored by Git

No coordinates or sequences in this directory come from the study datasets.

## 🚀 Run

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

Inspect the JSON summary printed by the command and verify that the generated NPZ contains balanced labels. Model training is optional and is documented in the main [README](../README.md).
