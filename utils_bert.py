import torch 
import torch.nn as nn 
import numpy as np 
import os
import argparse
import logging
import time
import numpy as np
import torch
import timeit
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
import math 
from torchdiffeq import odeint_adjoint as odeint
from PIL import Image
from tqdm import tqdm 

class BERT_feature_dataset(Dataset): 
    def __init__(self, x_np, y_np, transform=None, target_transform=None):
        self.x = torch.from_numpy(x_np).float()  
        self.y = torch.from_numpy(y_np).long()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,:]
        y = self.y[idx]
        
        # Apply transform to x
        if self.transform:
            x = self.transform(x)

        # Optional label transform
        if self.target_transform:
            y = self.target_transform(y)

        return x,y
    
def get_sst2_feature_dataset(path: str): 
    loaded_np = np.load(path)
    # Keys: train_feats, train_labels, val_feats, val_labels
    tr_ds = BERT_feature_dataset(loaded_np['train_feats'], loaded_np['train_labels'])
    val_ds = BERT_feature_dataset(loaded_np['val_feats'], loaded_np['val_labels'])

    print('Train DS: ', tr_ds.x.shape, tr_ds.y.shape)
    print('Val DS: ', val_ds.x.shape, val_ds.y.shape)
    
    return tr_ds, val_ds


def get_max_row_dist_for_2_classes(dim = 64):

    # make any random unit vector in R^64
    u = np.random.randn(dim)
    u /= np.linalg.norm(u)

    v1 = u
    v2 = -u

    W_binary_optimal = np.stack([v1, v2], axis=1)  # shape (64, 2)
    return W_binary_optimal

def check_max_row_dist_matrix(V, num_classes = 10):
    assert V.shape[1] == num_classes, 'V should be of shape [D, num_classes] '

    # --------------------------------------
    # 1. Check unit norms
    # --------------------------------------
    norms = np.linalg.norm(V, axis=0)
    print("Column norms:")
    print(norms)

    # --------------------------------------
    # 2. Compute cosine similarity matrix
    # --------------------------------------
    cos_sim = V.T @ V   # (10×10 matrix)
    n_classes = V.shape[1]
    desired_ip = 1.0 / (1.0 - n_classes)
    print(f"\nCosine similarity matrix (desired off diag={desired_ip}):")
    print(cos_sim)

    # --------------------------------------
    # 3. Extract off-diagonal values
    # --------------------------------------
    off_diag = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]

    # Print statistics
    print("\nOff-diagonal cosine similarity statistics:")
    print(f"mean: {off_diag.mean():.6f}")
    print(f"std : {off_diag.std():.6f}")
    print(f"min : {off_diag.min():.6f}")
    print(f"max : {off_diag.max():.6f}")

    # Optional: check if all similarities are equal (within tolerance)
    if np.allclose(off_diag, off_diag[0], atol=1e-4):
        print("\n✔ All off-diagonal similarities are equal.")
    else:
        print("\n✘ Off-diagonal similarities are NOT equal.")

import torch.nn as nn 
class BertCLF(nn.Module): 
    def __init__(self, dim_in, num_classes): 
        super(BertCLF, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(dim_in, num_classes)

    def forward(self, x): 
        return self.classifier(self.dropout(x))

def get_bert_fc_layer(path, dim_in = 768, num_classes = 2): 
    dummy_model = BertCLF(dim_in, num_classes)
    missing_keys, unexpected_keys = dummy_model.load_state_dict(torch.load(path), strict=False)
    print('missing_keys', missing_keys, 'len unexpected_keys', len(unexpected_keys))

    return dummy_model



def train_ce(epoch, model, loader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate((loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs

        outputs = model(x)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        'model': model, 
        'loss': train_loss/(batch_idx+1), 
        'acc': correct/total
    }


def test_ce(epoch, model, loader, device, criterion, best_acc, save_folder):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((loader)):
            print(inputs.shape, targets.shape)

            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = model(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            print(outputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_loss = test_loss/(batch_idx+1)
    acc = correct/total

    # Save checkpoint.
    acc = correct/total
    if acc > best_acc:
        print(f'Saving at epoch {epoch} with acc {acc} ...')
        state = {
            'phase1_model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, save_folder+'/phase1_best_acc_ckpt.pth')
        best_acc = acc
    
    return {
        'model': model,
        'loss': avg_loss, 
        'acc': acc,
        'best_acc': best_acc
    }
