# AttentionTAD

AttentionTAD classifies fixed-length DNA sequence windows as TAD-boundary or non-boundary regions. This repository contains the preprocessing and model-training code, together with a small synthetic example. Study datasets and reference genomes are not distributed here.

## Installation

AttentionTAD supports Python 3.9 through 3.13. NumPy is used for preprocessing; TensorFlow and Matplotlib are optional training dependencies.

```bash
git clone https://github.com/Yin-Shen/AttentionTAD.git
cd AttentionTAD

python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

## Example

The example uses a synthetic BED file and a generated synthetic FASTA. It is intended only to check that the workflow runs correctly.

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

The files produced under `examples/generated/` are ignored by Git.

## Input data

The preprocessor expects a BED file and a FASTA file. BED coordinates are zero-based and half-open.

| Column | Description | Required |
| --- | --- | --- |
| 1 | Chromosome or FASTA record name | Yes |
| 2 | Start coordinate | Yes |
| 3 | End coordinate | Yes |
| 4 | Interval name | No |

The default window size is 10 kb. Shorter or longer intervals are centered in a 10 kb window. Chromosome names in the BED file must match the FASTA headers. Duplicate coordinates are removed, and overlapping positive windows after normalization are rejected so that sequence is not shared between data partitions.

## Preprocessing

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

Negative windows are sampled with the supplied random seed and do not overlap the positive windows or one another. The compressed NPZ file contains `X_train`, `X_test`, `y_train`, `y_test`, and preprocessing metadata.

## Training

```bash
python -m attentiontad.training \
  --data outputs/cell_line_dataset.npz \
  --output-dir outputs/cell_line_run \
  --epochs 100 \
  --batch-size 128 \
  --seed 42
```

The training command writes the best `.keras` checkpoint, training history in CSV format, an optional history plot, and test metrics in JSON format. Use `--cpu` to disable GPU execution.

The installed command names `attentiontad-preprocess` and `attentiontad-train` provide the same interfaces. The two scripts in the repository root are retained as compatibility wrappers:

```bash
python tad_boundary_preprocessor.py --help
python train_tad_model.py --help
```

## Model

The default model has the following structure:

```text
10,000 x 4 one-hot sequence
  -> Conv1D (256 filters, kernel size 19)
  -> global max pooling
  -> learned feature weighting
  -> dense layer (64 units)
  -> sigmoid output
```

The filter count, kernel size, and dense-layer width can be changed from the training command line.

## Repository structure

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
|   |-- test_preprocessing.py
|   `-- test_training_io.py
|-- pyproject.toml
|-- tad_boundary_preprocessor.py
`-- train_tad_model.py
```

## Tests

The preprocessing and training I/O tests do not require TensorFlow:

```bash
python -m unittest discover -s tests -v
```

## Data and citation

Complete TAD-boundary annotations, reference genomes, processed arrays, and model checkpoints are excluded from version control. The BED file under `examples/data/` is entirely synthetic and is not a benchmark dataset.

Citation information will be added when the associated manuscript is publicly available. Questions and reproducibility reports can be submitted through [GitHub Issues](https://github.com/Yin-Shen/AttentionTAD/issues).
