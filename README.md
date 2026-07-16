# AttentionTAD

_Sequence-based preprocessing and deep-learning training for TAD boundary classification._

---

## 🔬 Overview

AttentionTAD classifies fixed-length genomic windows as TAD-boundary or non-boundary regions from one-hot encoded DNA sequence. This repository provides a focused public implementation of the preprocessing and attention-guided convolutional training workflow.

The public repository is intentionally compact:

- Explicit BED, FASTA, output, and random-seed parameters
- Deterministic balanced generation of mutually non-overlapping negative windows
- Compressed `uint8` datasets for lower storage and memory use
- TensorFlow training with validation-based checkpointing and held-out test metrics
- A fully synthetic example for checking the workflow without distributing study datasets

## 🚀 Quick start

### Requirements

| Requirement | Purpose |
| --- | --- |
| Python 3.9 to 3.13 | Preprocessing and command-line tools |
| TensorFlow | Model training |
| NumPy | Sequence encoding and dataset storage |

### Installation

```bash
git clone https://github.com/Yin-Shen/AttentionTAD.git
cd AttentionTAD

python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install AttentionTAD:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

### Run the synthetic example

```bash
python examples/generate_example_reference.py

python -m attentiontad.preprocessing \
  --bed examples/data/example_boundaries.bed \
  --fasta examples/generated/example_genome.fa \
  --output examples/generated/example_dataset.npz \
  --negative-bed examples/generated/example_negative_windows.bed \
  --seed 42

python -m attentiontad.training \
  --data examples/generated/example_dataset.npz \
  --output-dir examples/generated/example_run \
  --epochs 2 \
  --batch-size 4 \
  --validation-split 0.25 \
  --cpu
```

The bundled BED file and generated FASTA are synthetic workflow fixtures. They are not biological training or benchmark data.

## 🧬 Input data

### Boundary BED

The preprocessor reads the first four BED columns:

| Column | Meaning | Required |
| --- | --- | --- |
| 1 | Chromosome or FASTA record name | Yes |
| 2 | Zero-based start coordinate | Yes |
| 3 | Half-open end coordinate | Yes |
| 4 | Interval identifier | No |

Intervals that are not exactly 10 kb are centered in a 10 kb window by default. Chromosome names must match the FASTA headers.

Exact duplicate coordinates are collapsed. After normalization, positive windows must not overlap; this keeps shared sequence from crossing training, validation, or test partitions.

### Reference FASTA

Provide a local reference FASTA that uses the same genome assembly and chromosome naming convention as the BED file. Reference genomes, complete boundary sets, processed arrays, and model checkpoints are excluded from version control by default.

## ⚙️ Commands

### Preprocess a dataset

```bash
python -m attentiontad.preprocessing \
  --bed path/to/boundaries.bed \
  --fasta path/to/reference.fa \
  --output outputs/cell_line_dataset.npz \
  --negative-bed outputs/cell_line_negative_windows.bed \
  --window-size 10000 \
  --test-size 0.2 \
  --seed 42
```

The compressed NPZ contains `X_train`, `X_test`, `y_train`, `y_test`, and compact preprocessing metadata.

### Train and evaluate

```bash
python -m attentiontad.training \
  --data outputs/cell_line_dataset.npz \
  --output-dir outputs/cell_line_run \
  --epochs 100 \
  --batch-size 128 \
  --seed 42
```

Training writes the best `.keras` model, a CSV history file, an optional training plot, and held-out test metrics in JSON format. Use `--cpu` to disable GPU execution.

After installation, the equivalent short commands `attentiontad-preprocess` and `attentiontad-train` are also available when the Python scripts directory is on `PATH`.

The legacy root scripts remain as thin command-line wrappers:

```bash
python tad_boundary_preprocessor.py --help
python train_tad_model.py --help
```

## 🧠 Model

The public model follows this sequence:

`10,000 x 4 one-hot sequence -> Conv1D -> global max pooling -> learned feature weighting -> dense classifier -> boundary probability`

The default configuration uses 256 convolutional filters, a kernel size of 19, a 64-unit hidden layer, and a sigmoid output. These values can be changed from the training command line.

## 📁 Repository layout

```text
AttentionTAD/
|-- examples/
|   |-- data/example_boundaries.bed
|   |-- generate_example_reference.py
|   `-- README.md
|-- src/attentiontad/
|   |-- model.py
|   |-- preprocessing.py
|   `-- training.py
|-- tests/
|   `-- test_preprocessing.py
|-- pyproject.toml
|-- tad_boundary_preprocessor.py
`-- train_tad_model.py
```

## 🧪 Validation

Run the dependency-light preprocessing tests with:

```bash
python -m unittest discover -s tests -v
```

The synthetic example validates file parsing, deterministic negative sampling, sequence encoding, compressed dataset generation, and command-line integration. Full model training requires the optional TensorFlow dependencies.

## 📌 Research use

AttentionTAD is research software for studying sequence-associated TAD boundary patterns. Keep the BED coordinates, FASTA assembly, preprocessing parameters, and random seed aligned when reproducing an analysis.

Citation details will be added when the associated manuscript is publicly available.

## 🤝 Contributing

Questions, reproducibility reports, and focused improvements are welcome through [GitHub Issues](https://github.com/Yin-Shen/AttentionTAD/issues). Please avoid attaching complete genomic datasets or model checkpoints to issues or pull requests.
