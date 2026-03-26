#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 关键：统一长度为 9984
TARGET_LENGTH = 9984

class TADDataset(Dataset):
    """TAD边界预测数据集"""
    def __init__(self, sequences, labels, dataset_name='train'):
        """
        Args:
            sequences: (n_samples, 10000, 4)
            labels: (n_samples,)
        """
        # 强制裁剪到 9984
        self.sequences = sequences[:, :TARGET_LENGTH, :]
        self.labels = labels
        self.dataset_name = dataset_name
        
        print(f"\n{dataset_name.upper()}数据集初始化:")
        print(f"  sequences shape: {self.sequences.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  正样本数量: {np.sum(labels)} ({np.mean(labels)*100:.2f}%)")
        print(f"  负样本数量: {len(labels) - np.sum(labels)} ({(1-np.mean(labels))*100:.2f}%)")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).transpose(0, 1)  # (4, 9984)
        label = torch.FloatTensor([self.labels[idx]])
        return sequence, label


def load_tad_data(data_path, val_ratio=0.1, random_state=42):
    print(f"{'='*60}")
    print(f"加载TAD边界预测数据: {data_path}")
    print(f"{'='*60}")
    
    data = np.load(data_path)
    X_train = data['X_train'][:, :TARGET_LENGTH, :]  # 裁剪
    X_test = data['X_test'][:, :TARGET_LENGTH, :]
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"\n原始数据统计 (已裁剪到 {TARGET_LENGTH}):")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  训练集正样本比例: {np.mean(y_train)*100:.2f}%")
    print(f"  测试集正样本比例: {np.mean(y_test)*100:.2f}%")
    
    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_ratio, 
        random_state=random_state,
        stratify=y_train
    )
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    print(f"\n数据划分完成:")
    print(f"  训练集大小: {len(y_train_split)} (正样本: {np.sum(y_train_split)})")
    print(f"  验证集大小: {len(y_val)} (正样本: {np.sum(y_val)})")
    print(f"  测试集大小: {len(y_test)} (正样本: {np.sum(y_test)})")
    
    return X_train_split, X_val, X_test, y_train_split, y_val, y_test


def create_tad_data_loaders(data_path, batch_size=32, val_ratio=0.1, random_state=42):
    X_train, X_val, X_test, y_train, y_val, y_test = load_tad_data(
        data_path, val_ratio, random_state
    )
    
    train_dataset = TADDataset(X_train, y_train, 'train')
    val_dataset = TADDataset(X_val, y_val, 'val')
    test_dataset = TADDataset(X_test, y_test, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, y_val, y_test