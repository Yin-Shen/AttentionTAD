#!/usr/bin/env python
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

from data_loader import create_tad_data_loaders
from layer import create


def test_model(
    data_path,
    model_path='./output_tad/checkpoint/best_model.pth',
    num_class=2,
    multi=['seq'],
    enc_dims=[512, 384, 128],
    dec_dims=[200, 200],
    embed_dim=128,
    n_embed=200,
    split=16,
    ema=True,
    e_loss_weight=0.25,
    mu=0.01,
    batch_size=32,
    gpu=0,
    outdir='./test_results/',
    val_ratio=0.1,
    random_state=42,
    dataset_name='GM12878'
):
    """
    测试模型并生成评估指标和曲线
    
    Args:
        data_path: 数据路径
        model_path: 最佳模型路径
        outdir: 输出目录
        dataset_name: 数据集名称，用于文件命名
    """
    os.makedirs(outdir, exist_ok=True)
    
    # 设置设备
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
        print(f'Using GPU: {gpu}')
    else:
        device = 'cpu'
        print('Using CPU')
    
    # 加载数据
    print('Loading data...')
    train_loader, valid_loader, test_loader, valid_label, test_label = create_tad_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
    # 创建模型
    print('Creating model...')
    channel1, channel2, channel3 = enc_dims
    channel4, channel5 = dec_dims
    
    clf = create(
        num_class=num_class,
        multi=multi,
        channel1=channel1,
        channel2=channel2,
        channel3=channel3,
        channel4=channel4,
        channel5=channel5,
        embed_dim=embed_dim,
        n_embed=n_embed,
        split=split,
        ema=ema,
        e_loss_weight=e_loss_weight,
        mu=mu
    )
    
    # 加载模型权重
    print(f'Loading model from {model_path}...')
    checkpoint = torch.load(model_path, map_location=device)
    clf.load_state_dict(checkpoint['model'])
    clf = clf.to(device)
    clf.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model valid AUC: {checkpoint['valid_auc']:.6f}")
    print(f"Model test AUC: {checkpoint['test_auc']:.6f}")
    
    # 在测试集上评估
    print('\nEvaluating on test set...')
    test_out = []
    test_labels = []
    
    with torch.no_grad():
        for b_x0, b_y in tqdm(test_loader, desc='Testing'):
            b_x0 = b_x0.to(device)
            output, x0, _, _, _, _ = clf(b_x0)
            test_out.extend(x0.cpu().numpy())
            test_labels.extend(b_y.squeeze().cpu().numpy())
    
    test_out = np.array(test_out)
    test_labels = np.array(test_labels)
    
    # 获取预测概率和标签
    test_pred_prob = test_out[:, 1]  # 正类概率
    test_pred = (test_pred_prob > 0.5).astype(int)
    
    # 计算指标
    print('\n' + '='*60)
    print('Test Set Metrics:')
    print('='*60)
    
    accuracy = metrics.accuracy_score(test_labels, test_pred)
    auc = metrics.roc_auc_score(test_labels, test_pred_prob)
    aupr = metrics.average_precision_score(test_labels, test_pred_prob)
    recall = metrics.recall_score(test_labels, test_pred)
    precision = metrics.precision_score(test_labels, test_pred)
    f1 = metrics.f1_score(test_labels, test_pred)
    
    print(f'Accuracy:  {accuracy:.6f}')
    print(f'AUC:       {auc:.6f}')
    print(f'AUPR:      {aupr:.6f}')
    print(f'Recall:    {recall:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'F1-Score:  {f1:.6f}')
    print('='*60 + '\n')
    
    # 保存指标到文件
    metrics_file = os.path.join(outdir, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f'{dataset_name} Test Metrics\n')
        f.write('='*60 + '\n')
        f.write(f'Accuracy:  {accuracy:.6f}\n')
        f.write(f'AUC:       {auc:.6f}\n')
        f.write(f'AUPR:      {aupr:.6f}\n')
        f.write(f'Recall:    {recall:.6f}\n')
        f.write(f'Precision: {precision:.6f}\n')
        f.write(f'F1-Score:  {f1:.6f}\n')
        f.write('='*60 + '\n')
    
    print(f'Metrics saved to {metrics_file}')
    
    # 计算 ROC 曲线
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_labels, test_pred_prob)
    
    # 保存 ROC 数据
    roc_file = os.path.join(outdir, 'AUROC.txt')
    with open(roc_file, 'w') as f:
        f.write(f'CHINN-{dataset_name}-{dataset_name} ROC Curve Data\n')
        f.write(f'ROC AUC Score: {auc:.4f}\n\n')
        f.write('X-axis: False Positive Rate (FPR)\n')
        f.write('Y-axis: True Positive Rate (TPR)\n\n')
        f.write('FPR TPR\n')
        for fpr_val, tpr_val in zip(fpr, tpr):
            f.write(f'{fpr_val:.6f} {tpr_val:.6f}\n')
    
    print(f'ROC curve data saved to {roc_file}')
    
    # 计算 PR 曲线
    precision_curve, recall_curve, pr_thresholds = metrics.precision_recall_curve(test_labels, test_pred_prob)
    
    # 保存 PRC 数据
    prc_file = os.path.join(outdir, 'PRC.txt')
    with open(prc_file, 'w') as f:
        f.write(f'CHINN-{dataset_name}-{dataset_name} PR Curve Data\n')
        f.write(f'PR AUC Score: {aupr:.4f}\n\n')
        f.write('X-axis: Recall\n')
        f.write('Y-axis: Precision\n\n')
        f.write('Recall Precision\n')
        for rec_val, prec_val in zip(recall_curve, precision_curve):
            f.write(f'{rec_val:.6f} {prec_val:.6f}\n')
    
    print(f'PR curve data saved to {prc_file}')
    
    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{dataset_name} ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_file = os.path.join(outdir, 'ROC_curve.png')
    plt.savefig(roc_plot_file, dpi=300)
    plt.close()
    print(f'ROC curve plot saved to {roc_plot_file}')
    
    # 绘制 PR 曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, linewidth=2, label=f'PR curve (AUPR = {aupr:.4f})')
    baseline = np.sum(test_labels) / len(test_labels)
    plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline (Precision = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'{dataset_name} Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    prc_plot_file = os.path.join(outdir, 'PRC_curve.png')
    plt.savefig(prc_plot_file, dpi=300)
    plt.close()
    print(f'PR curve plot saved to {prc_plot_file}')
    
    # 保存混淆矩阵
    cm = metrics.confusion_matrix(test_labels, test_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{dataset_name} Confusion Matrix', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], fontsize=12)
    plt.yticks(tick_marks, ['Negative', 'Positive'], fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    cm_plot_file = os.path.join(outdir, 'confusion_matrix.png')
    plt.savefig(cm_plot_file, dpi=300)
    plt.close()
    print(f'Confusion matrix saved to {cm_plot_file}')
    
    # 保存预测结果
    np.save(os.path.join(outdir, 'test_predictions.npy'), test_pred)
    np.save(os.path.join(outdir, 'test_probabilities.npy'), test_pred_prob)
    np.save(os.path.join(outdir, 'test_labels.npy'), test_labels)
    
    print('\n' + '='*60)
    print('Testing completed! All results saved.')
    print('='*60)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'aupr': aupr,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }


if __name__ == '__main__':
    # 配置参数（需要与训练时保持一致）
    #data_path = '/public/home/shenyin/deep_learning/AttentionTAD/GM12878/GM12878_tad_boundary_data.npz'
    data_path = '/public/home/shenyin/deep_learning/AttentionTAD/Figure/GM12878_manually_independent_datas.npz'
    model_path = './output_tad/checkpoint/best_model.pth'
    
    # 运行测试
    results = test_model(
        data_path=data_path,
        model_path=model_path,
        num_class=2,
        multi=['seq'],
        enc_dims=[512, 384, 128],
        dec_dims=[200, 200],
        embed_dim=128,
        n_embed=200,
        split=16,
        ema=True,
        e_loss_weight=0.25,
        mu=0.01,
        batch_size=32,
        gpu=0,
        outdir='./Independent_test_results/',
        val_ratio=0.1,
        random_state=42,
        dataset_name='GM12878'
    )