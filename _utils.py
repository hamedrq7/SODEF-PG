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

### Model related
class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


class ODEfunc_mlp(nn.Module): 

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 64)
        self.act1 = torch.sin 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out
    
class MLP_OUT_LINEAR(nn.Module):
    def __init__(self):
        super(MLP_OUT_LINEAR, self).__init__()
        self.fc0 = nn.Linear(64, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_BALL(nn.Module):
    def __init__(self):
        super(MLP_OUT_BALL, self).__init__()
        self.fc0 = nn.Linear(64, 10, bias=False)
        fc_max = './EXP/fc_maxrowdistance_64_10/ckpt.pth'
        saved_temp = torch.load(fc_max)
        matrix_temp = saved_temp['matrix']
        self.fc0.weight.data = matrix_temp
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  
        


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)
      
class MLP_OUT_ORTH1024(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORTH1024, self).__init__()
        self.fc0 = ORTHFC(1024, 64, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

#####

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def accuracy(model, dataset_loader, device):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



        
def save_training_feature(model, dataset_loader, device, train_savepath):
    x_save = []
    y_save = []
    modulelist = list(model)
#     print(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
#         print(x.shape)
        
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
#     print(x_save.shape)
    
    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader, device, test_savepath):
    x_save = []
    y_save = []
    modulelist = list(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
    
    np.savez(test_savepath, x_save=x_save, y_save=y_save)
    

def get_loaders(dir_, batch_size, DATASET='CIFAR10'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    num_workers = 8

    if DATASET == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
        train_dataset__ = datasets.CIFAR10(
            dir_, train=True, transform=test_transform, download=True)
    elif DATASET == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_loader__ = torch.utils.data.DataLoader(
        dataset=train_dataset__,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader, train_loader__, test_dataset

import sys 
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f