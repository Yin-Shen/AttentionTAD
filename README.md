# AttentionTAD: Deep Learning Framework for TAD Boundary Prediction

## Introduction
AttentionTAD is a deep learning framework that integrates convolutional neural networks with attention mechanisms to predict topologically associating domain (TAD) boundaries directly from DNA sequences. The model achieves superior performance compared to existing methods while providing interpretable insights into the sequence determinants of three-dimensional genome organization. AttentionTAD demonstrates robust cross-cell-line generalizability and identifies biologically relevant transcription factor binding motifs associated with chromatin architecture.

## Key Features
- **Superior Performance**: Achieves 88.7% accuracy, 0.885 AUC, and 0.842 AUPR on independent test sets
- **Sequence-Only Input**: Requires only DNA sequence information without Hi-C data
- **Cell Type-Specific Predictions**: Supports 6 human cell lines (GM12878, HCT116, HMEC, HUVEC, K562, NHEK)
- **Cross-Cell-Line Generalizability**: Maintains discrimination abilities of 0.87-0.90 across diverse cellular contexts
- **Interpretable Results**: Attention mechanism reveals important sequence features and transcription factor motifs
- **Motif Discovery**: Identifies biologically relevant binding sites for CTCF, ZNF143, and MAZ
- **Web Interface**: Provides user-friendly online tool for TAD boundary prediction

## Steps to Install and Run AttentionTAD

### 1. Clone the AttentionTAD repository:
```
git clone https://github.com/yourusername/AttentionTAD.git
cd AttentionTAD
```

### 2. Install the required dependencies:
```
tensorflow>=2.4.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
biopython>=1.78
pybedtools>=0.8.0
matplotlib>=3.3.0
```

### 3. Prepare your input data:
Download the human reference genome (hg19.fa) and place it in the project directory

The repository includes TAD boundary BED files for seven cell lines:
- GM12878_TAD_boundaries.bed
- HCT116_TAD_boundaries.bed  
- HMEC_TAD_boundaries.bed
- HUVEC_TAD_boundaries.bed
- IMR90_TAD_boundaries.bed
- K562_TAD_boundaries.bed
- NHEK_TAD_boundaries.bed

### 4. Preprocess the data:
```
python tad_boundary_preprocessor.py GM12878
```

### 5. Train the AttentionTAD model:
```
python train_tad_model.py GM12878
```

### 6. Evaluate the trained model:
The training script automatically evaluates the model and generates performance plots and metrics in the output directory.

## Web Interface
AttentionTAD is also available as a web interface at: http://attentiontad.shenlabahmu.com

The web server provides cell type-specific TAD boundary predictions without requiring computational expertise or local installation.

## Model Architecture
AttentionTAD consists of:
- Input layer accepting 10,000 bp one-hot encoded DNA sequences
- Convolutional layer with 256 filters and kernel size 19
- Global max pooling layer
- Attention mechanism with two dense layers
- Fully connected layer with 64 neurons
- Output layer with sigmoid activation for boundary probability

## Performance Comparison
| Method | Accuracy | AUC | AUPR | Recall |
|--------|----------|-----|------|--------|
| AttentionTAD | 88.7% | 0.885 | 0.842 | 90.9% |
| pTADs | 73.0% | 0.537 | - | - |
| preciseTAD | 66.6% | 0.535 | - | - |
| TADBoundaryDetector | 25.5% | 0.629 | - | 21.1% |

## File Structure
```
AttentionTAD/
├── README.md
├── requirements.txt
├── tad_boundary_preprocessor.py
├── train_tad_model.py
├── GM12878_TAD_boundaries.bed
├── HCT116_TAD_boundaries.bed
├── HMEC_TAD_boundaries.bed
├── HUVEC_TAD_boundaries.bed
├── IMR90_TAD_boundaries.bed
├── K562_TAD_boundaries.bed
├── NHEK_TAD_boundaries.bed
└── hg19.fa (download separately)
```

## Citation
If you use AttentionTAD in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## Contact
For questions or support, please visit our web interface at http://attentiontad.shenlabahmu.com or contact the authors.

For more details on the AttentionTAD model architecture, training process, and evaluation metrics, please refer to the original publication.
