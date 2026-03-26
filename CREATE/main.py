#!/usr/bin/env python
import os
import numpy as np
import torch

from data_loader import create_tad_data_loaders
from create_train import CREATE_train
from layer import *
from logger import create_logger

import warnings
warnings.filterwarnings("ignore")

def CREATE(
        data_path, 
        num_class=2,
        multi=['seq'],
        test_aug=1, 
        train_aug=[1]*2,
        stride=10, 
        batch_size=32,
        enc_dims=[512, 384, 128], 
        dec_dims=[200, 200], 
        embed_dim=128, 
        n_embed=200, 
        split=16, 
        ema=True, 
        e_loss_weight=0.25, 
        mu=0.01, 
        open_loss_weight=0.01, 
        loop_loss_weight=0.1, 
        lr=5e-5, 
        max_epoch=300, 
        pre_epoch=50, 
        seed=0, 
        gpu=0, 
        outdir='./output/',
        val_ratio=0.1,
        random_state=42,
    ):
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'/log.txt')

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device='cuda'
        torch.cuda.set_device(gpu)
        log.info('Using GPU: {}'.format(gpu))
    else:
        device='cpu'
        log.info('Using CPU')
    
    log.info('Loading TAD boundary prediction data...')
    train_loader, valid_loader, test_loader, valid_label, test_label = create_tad_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
    # 修复：直接使用 enc_dims，不重新计算 channel1
    channel1, channel2, channel3 = enc_dims  # [512, 384, 128]
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
    
    log.info('Model architecture:')
    log.info(f'  Encoder dims: {enc_dims}')
    log.info(f'  Decoder dims: {dec_dims}')
    log.info(f'  Embed dim: {embed_dim}, n_embed: {n_embed}, split: {split}')
    log.info(f'  Number of parameters: {sum(p.numel() for p in clf.parameters())}')
    log.info('Model training...')

    CREATE_train(
        clf, 
        train_loader, 
        valid_loader, 
        test_loader, 
        valid_label, 
        test_label, 
        lr=lr, 
        max_epoch=max_epoch, 
        pre_epoch=pre_epoch, 
        multi=multi, 
        aug=test_aug*2,
        cls=num_class, 
        open_loss_weight=open_loss_weight, 
        loop_loss_weight=loop_loss_weight, 
        outdir=outdir,
        device=device
    )


if __name__ == '__main__':
    data_path = '/public/home/shenyin/deep_learning/AttentionTAD/GM12878/GM12878_tad_boundary_data.npz'
    
    CREATE(
        data_path=data_path,
        num_class=2,
        multi=['seq'],
        batch_size=32,
        lr=1e-4,
        max_epoch=100,
        pre_epoch=20,
        seed=42,
        gpu=0,
        outdir='./output_tad/',
        val_ratio=0.1,
    )