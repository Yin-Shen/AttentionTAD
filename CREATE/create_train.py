#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics

from layer import *
from utils import *


def CREATE_train(
        clf, 
        train_loader, 
        valid_loader, 
        test_loader, 
        valid_label, 
        test_label, 
        lr=5e-5, 
        max_epoch=300, 
        pre_epoch=50, 
        multi=['seq'], 
        aug=2, 
        cls=2,
        open_loss_weight=0.01, 
        loop_loss_weight=0.1, 
        outdir='./output/', 
        device='cuda'
    ):
    clf = clf.to(device)
    
    valid_labels = valid_label.reshape(-1, 1).astype(int)
    test_labels = test_label.reshape(-1, 1).astype(int)
    
    ids = {'seq':[],'open':[],'loop':[]}
    dicts = {'seq':4,'open':1,'loop':1}
    last = 0
    for omic in multi:
        ids[omic] = [last, last+dicts[omic]]
        last = last+dicts[omic]
    
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-8)
    
    clf_loss_fn = nn.BCEWithLogitsLoss()
    recon_loss1_ = nn.BCELoss()
    recon_loss2_ = nn.MSELoss()
    regloss = Regularization(clf, weight_decay1=1e-8, weight_decay2=5e-7)

    loss1_rate = [0.0] * pre_epoch + [1.0] * (max_epoch - pre_epoch)
    loss2_rate = [1.0] * pre_epoch + [0.5] * (max_epoch - pre_epoch)
    
    max_auc = 0.0
    max_auc_epoch = 0
    
    best_metrics = {
        'epoch': 0, 'train_acc': 0, 'valid_acc': 0, 'test_acc': 0,
        'train_auc': 0, 'valid_auc': 0, 'test_auc': 0,
        'train_aupr': 0, 'valid_aupr': 0, 'test_aupr': 0,
    }
    
    with tqdm(range(max_epoch), total=max_epoch, desc='Epochs') as tq:
        for epoch in tq:
            train_out, train_label_list, valid_out, test_out = [], [], [], []
            training_loss, training_clf, training_recon, training_latent, training_perplexity = [], [], [], [], []
            
            clf.train()
            for batch_idx, (b_x0, b_y) in enumerate(train_loader):
                train_label_list.extend(b_y.squeeze().cpu().numpy().tolist())
                b_x0 = b_x0.to(device)
                b_y = b_y.to(device).float()
                
                output, x0, out, _, latent_loss, perplexity3 = clf(b_x0)
                train_out.extend(x0.cpu().detach().numpy())

                reg_loss = regloss(clf)
                pred_prob = x0[:, 1:2]
                clf_loss = clf_loss_fn(output[:, 1:2], b_y) + reg_loss
                
                recon_loss = 0.0 * reg_loss
                if 'seq' in multi:
                    i1, i2 = ids['seq']
                    recon_loss += recon_loss1_(out[:, i1:i2], b_x0[:, i1:i2])
                if 'open' in multi:
                    i1, i2 = ids['open']
                    recon_loss += open_loss_weight * recon_loss2_(out[:, i1:i2], b_x0[:, i1:i2])
                if 'loop' in multi:
                    i1, i2 = ids['loop']
                    recon_loss += loop_loss_weight * recon_loss2_(out[:, i1:i2], b_x0[:, i1:i2])
                
                loss = loss1_rate[epoch] * (clf_loss + reg_loss) + loss2_rate[epoch] * (recon_loss + latent_loss)
                
                training_clf.append(torch.mean(clf_loss).item())
                training_recon.append(torch.mean(recon_loss).item())
                training_latent.append(torch.mean(latent_loss).item())
                training_perplexity.append(torch.mean(perplexity3).item())
                training_loss.append(torch.mean(loss).item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss = float(np.mean(training_loss))
            train_recon = float(np.mean(training_recon))
            train_latent = float(np.mean(training_latent))
            train_clf = float(np.mean(training_clf))
            train_perplexity = float(np.mean(training_perplexity))
            
            train_label_array = np.array(train_label_list)
            train_out_array = np.array(train_out)
            train_pred_prob = train_out_array[:, 1]
            train_pred = (train_pred_prob > 0.5).astype(int)
            
            epoch_info = 'recon=%.3f, latent=%.3f, perplexity=%.3f' % (train_recon, train_latent, train_perplexity) if epoch < pre_epoch else 'recon=%.3f, latent=%.3f, clf=%.3f, perplexity=%.3f' % (train_recon, train_latent, train_clf, train_perplexity)
            tq.set_postfix_str(epoch_info)

            clf.eval()
            with torch.no_grad():
                for batch_idx, (b_x0, b_y) in enumerate(valid_loader):
                    output, x0, _, _, _, _ = clf(b_x0.to(device))
                    valid_out.extend(x0.cpu().detach().numpy())
                
                valid_out_array = np.array(valid_out)
                valid_pred_prob = valid_out_array[:, 1]
                valid_pred = (valid_pred_prob > 0.5).astype(int)
                
                valid_acc = metrics.accuracy_score(valid_label, valid_pred)
                valid_auc = metrics.roc_auc_score(valid_label, valid_pred_prob)
                valid_aupr = metrics.average_precision_score(valid_label, valid_pred_prob)
                
                for batch_idx, (b_x0, b_y) in enumerate(test_loader):
                    output, x0, _, _, _, _ = clf(b_x0.to(device))
                    test_out.extend(x0.cpu().detach().numpy())
                
                test_out_array = np.array(test_out)
                test_pred_prob = test_out_array[:, 1]
                test_pred = (test_pred_prob > 0.5).astype(int)

            if epoch >= pre_epoch:
                train_acc = metrics.accuracy_score(train_label_array, train_pred)
                train_auc = metrics.roc_auc_score(train_label_array, train_pred_prob)
                train_aupr = metrics.average_precision_score(train_label_array, train_pred_prob)
                
                test_acc = metrics.accuracy_score(test_label, test_pred)
                test_auc = metrics.roc_auc_score(test_label, test_pred_prob)
                test_aupr = metrics.average_precision_score(test_label, test_pred_prob)
                
                print(f"\nEpoch {epoch+1}/{max_epoch}")
                print(f"Train - Acc: {train_acc:.6f}, AUC: {train_auc:.6f}, AUPR: {train_aupr:.6f}")
                print(f"Valid - Acc: {valid_acc:.6f}, AUC: {valid_auc:.6f}, AUPR: {valid_aupr:.6f}")
                print(f"Test  - Acc: {test_acc:.6f}, AUC: {test_auc:.6f}, AUPR: {test_aupr:.6f}")

            if epoch < pre_epoch:
                scheduler1.step(train_recon + train_latent)
            else:
                scheduler2.step(train_loss)

            if valid_auc > max_auc:
                max_auc = valid_auc
                max_auc_epoch = epoch
                
                if epoch >= pre_epoch - 1:
                    np.save(outdir+'best_test_prob.npy', test_pred_prob)
                    np.save(outdir+'best_test_pred.npy', test_pred)
                    
                    state = {
                        'model': clf.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'epoch': epoch + 1,
                        'valid_auc': valid_auc,
                        'test_auc': test_auc,
                    }
                    torch.save(state, outdir+'checkpoint/best_model.pth')
                    
                    if epoch >= pre_epoch:
                        best_metrics.update({
                            'epoch': epoch + 1, 'train_acc': train_acc, 'valid_acc': valid_acc, 'test_acc': test_acc,
                            'train_auc': train_auc, 'valid_auc': valid_auc, 'test_auc': test_auc,
                            'train_aupr': train_aupr, 'valid_aupr': valid_aupr, 'test_aupr': test_aupr,
                        })
                    
                    print(f"\n{'='*60}")
                    print(f"New best model saved! Valid AUC: {valid_auc:.6f}")
                    print(f"{'='*60}\n")
            
            if epoch - max_auc_epoch >= 30 and epoch >= 100:
                print("\n" + "="*60)
                print("Early stopping triggered!")
                print("="*60)
                break

        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        print(f"\nBest model at epoch {best_metrics['epoch']}:")
        print(f"Train - Acc: {best_metrics['train_acc']:.6f}, AUC: {best_metrics['train_auc']:.6f}, AUPR: {best_metrics['train_aupr']:.6f}")
        print(f"Valid - Acc: {best_metrics['valid_acc']:.6f}, AUC: {best_metrics['valid_auc']:.6f}, AUPR: {best_metrics['valid_aupr']:.6f}")
        print(f"Test  - Acc: {best_metrics['test_acc']:.6f}, AUC: {best_metrics['test_auc']:.6f}, AUPR: {best_metrics['test_aupr']:.6f}")
        print("="*60 + "\n")
        
        torch.cuda.empty_cache()