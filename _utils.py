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

### Model related

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value    



class ODEBlocktemp(nn.Module):  ####  note here we do not integrate to save time

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


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
    def __init__(self, path: str = './EXP/fc_maxrowdistance_64_10/ckpt.pth'):
        super(MLP_OUT_BALL, self,).__init__()
        self.fc0 = nn.Linear(64, 10, bias=False)
        saved_temp = torch.load(path, map_location=torch.device('cpu') )
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
    
class MLP_OUT_ORTH512(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORTH512, self).__init__()
        self.fc0 = ORTHFC(512, 64, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
#####

def test_phase3(net, epoch, testloader, criterion, device, folder_savemodel):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
#     modulelist = list(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = net(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, folder_savemodel+'/ckpt.pth')
        best_acc = acc
        

# Training
def train_phase3(net, epoch, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
#         print(inputs.shape)

        outputs = net(x)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

def df_dz_regularizer(f, z, numm, odefunc, time_df, exponent, trans, exponent_off, transoffdig, device):
#     print("+++++++++++")
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1],-1)
        if batchijacobian.shape[0]!=batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")
            
        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent*(tempdiag+trans))
        offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
        off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
        regu_offdiag += off_diagtemp

    print('diag mean: ',tempdiag.mean().item())
    print('offdiag mean: ',offdiat.mean().item())
    return regu_diag/numm, regu_offdiag/numm

def f_regularizer(f, z, odefunc, time_df, device, exponent_f):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    print('tempf: ', tempf.mean().item())
    
    return regu_f


def temp1(f, z, text_file, numm, odefunc, time_df, device, exponent, trans, exponent_off, transoffdig):
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1],-1)
        if batchijacobian.shape[0]!=batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")
        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent*(tempdiag+trans))
        offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
        off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
        regu_offdiag += off_diagtemp

    text_file.write('diag mean: '+str(tempdiag.mean().item())+'\n')
    text_file.write('offdiag mean: '+str(offdiat.mean().item())+'\n')
    return 0
def temp2(f, z, text_file, odefunc, time_df, device, exponent_f):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    text_file.write('tempf: '+str(tempf.mean().item())+'\n')
    return 0


class DensemnistDatasetTrain(Dataset):
    def __init__(self, train_savepath):
        """
        """
        npzfile = np.load(train_savepath)
        
        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y
class DensemnistDatasetTest(Dataset):
    def __init__(self, test_savepath):
        """
        """
        npzfile = np.load(test_savepath)
        
        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y    

def train_save_robustfeature(epoch, model, loader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
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

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

def test_save_robustfeature(epoch, model, loader, device, criterion, best_acc, save_folder, train_eval_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = model(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net_save_robustfeature': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, save_folder+'/ckpt.pth')
        best_acc = acc
        
        train_savepath='./data/CIFAR10_train_resnetNov1.npz'
        test_savepath='./data/CIFAR10_test_resnetNov1.npz'
    
        save_training_feature(model, train_eval_loader, device, train_savepath)
        print('----')
        save_testing_feature(model, loader, device, test_savepath)
        print('------------')
    
    return best_acc

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
    num_workers = 2

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



def evaluate_standard(test_loader, model, device):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            output = model((X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n


upper_limit, lower_limit = 1, 0

# cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_mean = (0.0, 0.0, 0.0)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, eps=8, step=2, use_CWloss=False):
    epsilon = eps / 255.
    alpha = step / 255.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=use_CWloss)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n
