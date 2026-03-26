import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os
from tqdm import tqdm
import warnings
import logging
from Bio import SeqIO
import matplotlib

# 导入你的模型相关模块
from layer import create
from data_loader import TADDataset
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    plt.style.use('seaborn-paper')
except:
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('default')

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

REFERENCE_GENOME = "/public/home/shenyin/deep_learning/AttentionTAD/hg38.fa"
TAD_BED_FILE = "/public/home/shenyin/deep_learning/AttentionTAD/GSM4847912.contact_domains.bed"
BOUNDARY_SIZE = 10000
TARGET_LENGTH = 9984

def read_fasta(fasta_file):
    """读取FASTA文件"""
    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fasta_file, "fasta")}

def extract_tad_boundaries(bed_file, boundary_size=BOUNDARY_SIZE):
    """从BED文件提取TAD边界"""
    positive_samples = []
    tads = []
    
    with open(bed_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            tads.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'name': f"{chrom}:{start}-{end}"
            })
    
    for i in range(1, len(tads)):
        curr_tad = tads[i]
        prev_tad = tads[i-1]
        
        if curr_tad['chrom'] == prev_tad['chrom']:
            mid_point = (prev_tad['end'] + curr_tad['start']) // 2
            boundary_start = mid_point - boundary_size // 2
            boundary_end = mid_point + boundary_size // 2
            
            boundary_start = max(0, boundary_start)
            
            positive_samples.append({
                'chrom': curr_tad['chrom'],
                'start': boundary_start,
                'end': boundary_end,
                'name': f"{curr_tad['chrom']}:{boundary_start}-{boundary_end}",
                'label': 1
            })
    
    logger.info(f"Extracted {len(positive_samples)} positive samples (TAD boundaries) from {bed_file}")
    return positive_samples

def ensure_10kb_region(chrom, start, end, genome_dict):
    """确保区域为10kb"""
    length = end - start
    
    if length == 10000:
        return start, end, 0
    
    if length < 10000:
        extension = (10000 - length) // 2
        new_start = max(0, start - extension)
        
        chrom_length = len(genome_dict.get(chrom, ""))
        if chrom_length == 0:
            extension_end = extension
        else:
            start_extension_used = start - new_start
            extension_end = extension + (extension - start_extension_used)
            
        new_end = min(end + extension_end, chrom_length) if chrom_length > 0 else end + extension_end
        
        if chrom_length > 0 and new_end - new_start < 10000:
            new_start = max(0, new_start - (10000 - (new_end - new_start)))
            
        final_length = new_end - new_start
        if final_length < 10000:
            return new_start, new_end, 10000 - final_length
        
        return new_start, new_end, 0
        
    else:
        center = (start + end) // 2
        new_start = center - 5000
        new_end = center + 5000
        
        return new_start, new_end, 0

def one_hot_encode(sequence):
    """对DNA序列进行one-hot编码"""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = np.array([mapping.get(base.upper(), mapping['N']) for base in sequence])
    return encoded

def process_tad_boundary(chrom, start, end, genome_dict):
    """处理单个TAD边界"""
    if chrom not in genome_dict:
        if chrom.startswith('chr'):
            chrom_key = chrom[3:]
        else:
            chrom_key = f"chr{chrom}"
    else:
        chrom_key = chrom
    
    if chrom_key not in genome_dict:
        logger.warning(f"Chromosome {chrom} not found in genome")
        return None
    
    new_start, new_end, padding_needed = ensure_10kb_region(chrom_key, start, end, genome_dict)
    
    if new_start >= len(genome_dict[chrom_key]) or new_end > len(genome_dict[chrom_key]):
        logger.warning(f"Region {chrom}:{new_start}-{new_end} exceeds chromosome length")
        return None
    
    sequence = genome_dict[chrom_key][new_start:new_end]
    
    if padding_needed > 0:
        sequence = sequence + 'N' * padding_needed
    
    encoded_seq = one_hot_encode(sequence)
    return encoded_seq

def process_tad_boundaries(boundaries, genome_dict):
    """处理所有TAD边界"""
    X_data = []
    y_data = []
    names = []
    
    for boundary in boundaries:
        chrom = boundary['chrom']
        start = boundary['start']
        end = boundary['end']
        
        encoded_seq = process_tad_boundary(chrom, start, end, genome_dict)
        
        if encoded_seq is not None:
            X_data.append(encoded_seq)
            y_data.append(boundary['label'])
            names.append(boundary['name'])
    
    if X_data:
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        return X_data, y_data, names
    else:
        return None, None, None

def load_create_model(model_path, device='cuda'):
    """加载CREATE模型"""
    logger.info(f"Loading CREATE model from {model_path}")
    
    # 创建模型实例
    model = create(
        num_class=2,
        multi=['seq'],
        channel1=512,
        channel2=384,
        channel3=128,
        channel4=200,
        channel5=200,
        embed_dim=128,
        n_embed=200,
        split=16,
        ema=True,
        e_loss_weight=0.25,
        mu=0.01
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"CREATE model loaded successfully")
    logger.info(f"  Valid AUC: {checkpoint.get('valid_auc', 'N/A')}")
    logger.info(f"  Test AUC: {checkpoint.get('test_auc', 'N/A')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model

def predict_with_create(model, X_data, batch_size=32, device='cuda'):
    """使用CREATE模型进行预测"""
    # 裁剪到TARGET_LENGTH
    X_data = X_data[:, :TARGET_LENGTH, :]
    
    # 创建数据集和数据加载器
    y_dummy = np.zeros(len(X_data))
    dataset = TADDataset(X_data, y_dummy, 'predict')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch_x, _ in tqdm(dataloader, desc="Predicting"):
            batch_x = batch_x.to(device)
            _, x0, _, _, _, _ = model(batch_x)
            pred_prob = x0[:, 1].cpu().numpy()  # 取正类概率
            predictions.extend(pred_prob)
    
    return np.array(predictions)

def plot_metrics(results, output_dir):
    """绘制评估指标"""
    metrics_to_plot = [
        'Positive_Accuracy', 
        'Average_Confidence', 
        'Pct_HighConf_80',
        'Pct_HighConf_90', 
        'Pct_HighConf_95'
    ]
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4*len(metrics_to_plot)))
    
    for i, metric in enumerate(metrics_to_plot):
        value = results[metric]
        
        if metric.startswith('Pct_'):
            y_label = "Percentage (%)"
        elif metric == 'Positive_Accuracy':
            y_label = "Accuracy (%)"
            value = value * 100
        else:
            y_label = "Value"
            
        ax = axes[i] if len(metrics_to_plot) > 1 else axes
        bar = ax.bar(['CREATE'], [value], color=sns.color_palette("muted")[0])
        
        height = bar[0].get_height()
        ax.text(bar[0].get_x() + bar[0].get_width()/2., height + 0.01,
                f'{value:.2f}' if metric != 'Positive_Accuracy' else f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10)
        
        metric_display_name = metric.replace('_', ' ')
        ax.set_title(f"{metric_display_name}")
        ax.set_ylabel(y_label)
        ax.set_ylim(0, value * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("CREATE Model Performance on TAD Boundaries", fontsize=16, y=0.92)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/create_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/create_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(predictions, output_dir):
    """绘制置信度分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(predictions, bins=20, kde=True, ax=ax, color=sns.color_palette("muted")[0])
    
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
    
    stats_text = f"Mean: {np.mean(predictions):.3f}\n"
    stats_text += f"Median: {np.median(predictions):.3f}\n"
    stats_text += f"% ≥ 0.5: {np.mean(predictions >= 0.5)*100:.1f}%\n"
    stats_text += f"% ≥ 0.8: {np.mean(predictions >= 0.8)*100:.1f}%\n"
    stats_text += f"% ≥ 0.9: {np.mean(predictions >= 0.9)*100:.1f}%"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title("CREATE Model Confidence Distribution", fontsize=14)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/create_confidence_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/create_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_summary(results, predictions, names, output_dir):
    """保存结果摘要"""
    # 保存详细统计
    with open(f'{output_dir}/create_results_summary.txt', 'w') as f:
        f.write("==== CREATE Model - TAD Boundary Prediction Results ====\n\n")
        f.write(f"Total positive samples evaluated: {results['Positive_Sample_Count']}\n")
        f.write(f"Positive sample accuracy: {results['Positive_Accuracy']*100:.2f}%\n")
        f.write(f"Average confidence score: {results['Average_Confidence']:.4f}\n")
        f.write(f"Confidence score distribution:\n")
        f.write(f"  Min: {results['Confidence_Min']:.4f}\n")
        f.write(f"  25th percentile: {results['Confidence_25th']:.4f}\n")
        f.write(f"  Median: {results['Confidence_Median']:.4f}\n")
        f.write(f"  75th percentile: {results['Confidence_75th']:.4f}\n")
        f.write(f"  Max: {results['Confidence_Max']:.4f}\n")
        f.write(f"  Standard deviation: {results['Confidence_StdDev']:.4f}\n")
        f.write(f"Percentage of positive samples with high confidence:\n")
        f.write(f"  ≥ 0.80: {results['Pct_HighConf_80']:.2f}%\n")
        f.write(f"  ≥ 0.90: {results['Pct_HighConf_90']:.2f}%\n")
        f.write(f"  ≥ 0.95: {results['Pct_HighConf_95']:.2f}%\n")
        
        f.write("\nDetailed confidence score distribution:\n")
        for bin_range, percentage in results['Confidence_Distribution'].items():
            f.write(f"  {bin_range}: {percentage}\n")
    
    # 保存预测结果
    prediction_df = pd.DataFrame({
        'name': names,
        'CREATE_prediction': predictions,
        'CREATE_class': (predictions > 0.5).astype(int)
    })
    prediction_df.to_csv(f'{output_dir}/create_predictions.csv', index=False)
    
    # 保存汇总CSV
    summary_df = pd.DataFrame([{
        'Model': 'CREATE',
        'Sample_Count': results['Positive_Sample_Count'],
        'Accuracy': results['Positive_Accuracy'] * 100,
        'Avg_Confidence': results['Average_Confidence'],
        'Median_Confidence': results['Confidence_Median'],
        'HighConf_80': results['Pct_HighConf_80'],
        'HighConf_90': results['Pct_HighConf_90'],
        'HighConf_95': results['Pct_HighConf_95']
    }])
    summary_df.to_csv(f'{output_dir}/create_summary.csv', index=False)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    cell_line = 'GM12878'
    output_dir = f'{cell_line}_create_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 加载CREATE模型
    model_path = f'{cell_line}/output/checkpoint/best_model.pth'
    if not os.path.exists(model_path):
        model_path = './output_tad/checkpoint/best_model.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    model = load_create_model(model_path, device=device)
    
    # 加载基因组和TAD边界
    logger.info("Loading genome reference...")
    genome_dict = read_fasta(REFERENCE_GENOME)
    
    logger.info("Extracting TAD boundaries from BED file...")
    positive_samples = extract_tad_boundaries(TAD_BED_FILE)
    
    logger.info("Processing boundary sequences...")
    X_data, y_data, names = process_tad_boundaries(positive_samples, genome_dict)
    
    if X_data is None:
        logger.error("Failed to process boundary sequences")
        return
    
    logger.info(f"Successfully processed {len(X_data)} boundary sequences")
    
    # 保存样本信息
    sample_df = pd.DataFrame({
        'name': names,
        'label': y_data
    })
    sample_df.to_csv(f'{output_dir}/all_positive_samples.tsv', sep='\t', index=False)
    
    # 使用CREATE模型预测
    logger.info("Evaluating CREATE model on TAD boundaries...")
    predictions = predict_with_create(model, X_data, batch_size=32, device=device)
    
    # 计算评估指标
    pred_binary = (predictions > 0.5).astype(int)
    
    results = {
        'Positive_Sample_Count': len(predictions),
        'Positive_Accuracy': np.mean(pred_binary),
        'Average_Confidence': np.mean(predictions),
        'Confidence_StdDev': np.std(predictions),
        'Confidence_Min': np.min(predictions),
        'Confidence_25th': np.percentile(predictions, 25),
        'Confidence_Median': np.median(predictions),
        'Confidence_75th': np.percentile(predictions, 75),
        'Confidence_Max': np.max(predictions),
        'Pct_HighConf_80': np.sum(predictions >= 0.8) / len(predictions) * 100,
        'Pct_HighConf_90': np.sum(predictions >= 0.9) / len(predictions) * 100,
        'Pct_HighConf_95': np.sum(predictions >= 0.95) / len(predictions) * 100,
        'Confidence_Distribution': {
            f"{i/10:.1f}-{(i+1)/10:.1f}": f"{np.sum((predictions >= i/10) & (predictions < (i+1)/10))/len(predictions)*100:.2f}%" 
            for i in range(10)
        }
    }
    
    # 保存和可视化结果
    save_results_summary(results, predictions, names, output_dir)
    plot_metrics(results, output_dir)
    plot_confidence_distribution(predictions, output_dir)
    
    # 打印结果
    print("\n" + "="*60)
    print("CREATE Model - TAD Boundary Prediction Results")
    print("="*60)
    print(f"Boundaries evaluated: {results['Positive_Sample_Count']}")
    print(f"Accuracy (>0.5 threshold): {results['Positive_Accuracy']*100:.2f}%")
    print(f"Average confidence score: {results['Average_Confidence']:.4f}")
    print(f"Median confidence score: {results['Confidence_Median']:.4f}")
    print(f"High confidence scores (≥0.8): {results['Pct_HighConf_80']:.2f}%")
    print(f"High confidence scores (≥0.9): {results['Pct_HighConf_90']:.2f}%")
    print(f"High confidence scores (≥0.95): {results['Pct_HighConf_95']:.2f}%")
    print("="*60 + "\n")
    
    logger.info(f"Evaluation completed! Results saved to {output_dir}")

if __name__ == "__main__":
    main()