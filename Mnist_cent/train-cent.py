from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import logging
from torch.nn.parameter import Parameter
import geotorch
import math
import random
import numpy as np
import os
import argparse
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import Dataset, DataLoader
from models import *
import wandb
from model import * 

cent_weight = 0.001
cent_lr = 0.0
rad = 20.0
exp_name = f'cw_{cent_weight}-clr_{cent_lr}-rad{rad}_no_bias'
do_wandb = True
# device = torch.device("cuda:0")
torch.cuda.set_device(1)
device = torch.device("cuda:1")
best_acc = 0
start_epoch = 0

if do_wandb:
    wandb.init(project="SODEF-MNIST", name=f'MNIST-64D-CenterLoss-FCinit_cent_weight_{exp_name}')

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

base_data_path = f'./data_cent_init_{exp_name}'
train_savepath = f'{base_data_path}/orth_MNIST_train_resnet_final.npz'
test_savepath = f'{base_data_path}/orth_MNIST_test_resnet_final.npz'
os.makedirs(base_data_path, exist_ok=True)

fc_dim = 64
folder_savemodel = f'./EXP/orth_MNIST_resnet_final_{exp_name}'
os.makedirs(folder_savemodel, exist_ok=True)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


args = {} # get_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)





def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True
    )

    return train_loader, test_loader, train_eval_loader


    
trainloader, testloader, train_eval_loader = get_mnist_loaders(
    False, 128, 1000
)

print('==> Building model..')
from models import *

net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

net = net.to(device)

net = nn.Sequential(*list(net.children())[0:-1])

fcs_temp = MLP_OUT_ORTH512()
# fcs_temp = fcs()

fc_layers = MLP_OUT_BALL()
for param in fc_layers.parameters():
    param.requires_grad = False
net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)
from cent import CenterLossNormal
# centers = CenterLossNormal(10, feat_dim=64, use_gpu=True, init_value=fc_layers.fc0.weight.detach().clone()* 20.0)
centers = CenterLossNormal(10, feat_dim=64, init_value=fc_layers.fc0.weight.detach().clone()* rad)
centers = centers.to(device)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0001, 'eps':1e-4,},
                            {'params': centers.parameters(), 'lr': cent_lr, 'eps':1e-4,}], weight_decay=0.0005, amsgrad=True)


def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:6]:
            x = l(x)
        x = net[6](x[..., 0, 0])
        xo = x

        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())

        for l in modulelist[0:6]:
            x = l(x)
        x = net[6](x[..., 0, 0])
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)

    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)

    np.savez(test_savepath, x_save=x_save, y_save=y_save)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    modulelist = list(net)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs

        for l in modulelist[0:6]:
            x = l(x)
        feats = net[6](x[..., 0, 0])
        
        cent_loss = centers(feats, targets)
        outputs = net[7](feats)
        
        loss = criterion(outputs, targets) + cent_weight * cent_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx == 0:
            acc = 100. * correct / total
            if do_wandb:
                wandb.log({
                    'phase1/step': epoch,
                    'phase1/train_acc': acc,
                    'phase1/train_cent_loss': cent_loss.item(),
                })
            # break # [X]

        # print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    modulelist = list(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            for l in modulelist[0:6]:
                #             print(l)
                #             print(x.shape)
                x = l(x)

            x = net[6](x[..., 0, 0])
            cent_loss = centers(x, targets)
            x = net[7](x)
            outputs = x
            loss = criterion(outputs, targets) + cent_weight * cent_loss

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #       % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            if batch_idx == 0:
                acc = 100. * correct / total
                if do_wandb:
                    wandb.log({
                        'phase1/step': epoch,
                        'phase1/test_acc': acc,
                        'phase1/test_cent_loss': cent_loss.item(),
                    })
                # [X]

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'cent': centers.state_dict(),
        }
        #         if not os.path.isdir('checkpoint'):
        #             os.mkdir('checkpoint')
        #         torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, folder_savemodel + '/ckpt.pth')
        best_acc = acc

        save_training_feature(net, train_eval_loader)
        # print('----')
        save_testing_feature(net, testloader)
        # print('------------')

from tqdm import trange
############################################### Phase 1 ################################################
makedirs(folder_savemodel)
makedirs('./data')
for epoch in trange(0, 25):
    train(epoch)
    test(epoch)
    # break # [X]

# print('saved')
# phase1_ckpt = torch.load('/content/SODEF/mnist_resnet/orth_MNIST_resnet_final_ckpt.pth')['net']
# print(phase1_ckpt.keys())
# print(phase1_ckpt['6.fc0.linear.parametrizations.weight.fibr_aux'].shape)
# # print(phase1_ckpt['6.fc0.linear.parametrizations.weight.original'].shape)
# print(phase1_ckpt['6.fc0.linear.parametrizations.weight.base'].shape)

# print('created model')
# ss = net.state_dict()
# print(ss.keys())
# # print(ss['6.fc0.linear.parametrizations.weight.original'].shape)
# print(ss['6.fc0.linear.parametrizations.weight.0.base'].shape)
 
# net.load_state_dict(phase1_ckpt)
# print(net)

################################################ Phase 2 ################################################
weight_diag = 10
weight_offdiag = 10
weight_norm = 0
weight_lossc = 0
weight_f = 0.2

exponent = 1.0
exponent_f = 50
exponent_off = 0.1

endtime = 1

trans = 1.0
transoffdig = 1.0
trans_f = 0.0
numm = 8
timescale = 1
fc_dim = 64
t_dim = 1
act = torch.sin
act2 = torch.nn.functional.relu

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):  # dense_resnet_relu1,2,7

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.act1 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1 * self.fc1(t, x)
        out = self.act1(out)
        return out

class ODEBlocktemp(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class MLP_OUT(nn.Module):

    def __init__(self):
        super(MLP_OUT, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10, bias=False)

    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def df_dz_regularizer(f, z):
    #     print("+++++++++++")
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm, z.shape[0]), replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(1.0).to(device), x),
                                                            z[ii:ii + 1, ...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1], -1)
        if batchijacobian.shape[0] != batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")

        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent * (tempdiag + trans))
        #         print(regu_diag)

        offdiat = torch.sum(
            torch.abs(batchijacobian) * ((-1 * torch.eye(batchijacobian.shape[0]).to(device) + 0.5) * 2), dim=0)
        off_diagtemp = torch.exp(exponent_off * (offdiat + transoffdig))
        #         off_diagtemp = torch.exp(exponent*(torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)+transoffdig))
        regu_offdiag += off_diagtemp

    #     a = tempdiag<-0.000001
    #     aa = tempdiag[a]
    #     print('tempdiag dim  ', tempdiag.shape)
    # print('diag mean: ', tempdiag.mean().item())
    # print('offdiag mean: ', offdiat.mean().item())
    return regu_diag / numm, regu_offdiag / numm


def f_regularizer(f, z):
    tempf = torch.abs(odefunc(torch.tensor(1.0).to(device), z))
    regu_f = torch.pow(exponent_f * tempf, 2)
    #     regu_f = torch.exp(exponent_f*tempf+trans_f)
    #     regu_f = torch.log(tempf+1e-8)
    # print('tempf: ', tempf.mean().item())

    return regu_f


def critialpoint_regularizer(y1):
    regu4 = torch.linalg.norm(y1, dim=1)
    regu4 = regu4.mean()
    # print('regu4 norm: ', regu4)
    #     regu4 = torch.pow(regu4,2)
    regu4 = torch.exp(-0.1 * regu4 + 5)
    return regu4.mean()


class DensemnistDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y


class DensemnistDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)

        self.x = npzfile['x_save']
        self.y = npzfile['y_save']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx]

        return x, y

odesavefolder = f'./EXP/orth_dense_resnet_final_{exp_name}'
makedirs(odesavefolder)
odefunc = ODEfunc_mlp(0)

feature_layers = [ODEBlocktemp(odefunc)]
fc_layers = [MLP_OUT()]

for param in fc_layers[0].parameters():
    param.requires_grad = False

model = nn.Sequential(*feature_layers, *fc_layers).to(device)
criterion = nn.CrossEntropyLoss().to(device)
regularizer = nn.MSELoss()

train_loader = DataLoader(DensemnistDatasetTrain(),
                          batch_size=32,
                          shuffle=True, num_workers=4, pin_memory=True
                          )
train_loader__ = DataLoader(DensemnistDatasetTrain(),
                            batch_size=32,
                            shuffle=True, num_workers=4, pin_memory=True
                            )

test_loader = DataLoader(DensemnistDatasetTest(),
                         batch_size=32,
                         shuffle=True, num_workers=4, pin_memory=True
                         )

data_gen = inf_generator(train_loader)
val_data_gen = inf_generator(test_loader)
batches_per_epoch = len(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

best_acc = 0
tempi = 0

for itr in trange(40 * batches_per_epoch):
    # break

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)
    y = y.to(device)
    modulelist = list(model)
    y0 = x
    x = modulelist[0](x)
    y1 = x
    for l in modulelist[1:]:
        x = l(x)
    logits = x
    y00 = y0  # .clone().detach().requires_grad_(True)

    regu1, regu2 = df_dz_regularizer(odefunc, y00)
    regu1 = regu1.mean()
    regu2 = regu2.mean()
    # print("regu1:weight_diag " + str(regu1.item()) + ':' + str(weight_diag))
    # print("regu2:weight_offdiag " + str(regu2.item()) + ':' + str(weight_offdiag))
    regu3 = f_regularizer(odefunc, y00)
    regu3 = regu3.mean()
    # print("regu3:weight_f " + str(regu3.item()) + ':' + str(weight_f))
    loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2
    #         loss = weight_f*regu3

    if do_wandb:
        wandb.log({
            'phase2/reg1': regu1.item(),
            'phase2/reg2': regu2.item(),
            'phase2/reg3': regu3.item(),
            'phase2/loss': loss.item(),
        })

    if itr % 100 == 1:
        torch.save({'state_dict': model.state_dict(), 'args': args},
                   os.path.join(odesavefolder, 'model_diag.pth' + str(itr // 100)))
    # print("odesavefolder  ", odesavefolder)

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    # if itr % 50 == 0:
    #     val_data_gen = inf_generator(test_loader)
    #     for test_iter in range(20): # [X]
    #         with torch.no_grad():
    #             x, y = val_data_gen.__next__()
    #             x = x.to(device)
    #             y = y.to(device)
    #             modulelist = list(model)
    #             y0 = x
    #             x = modulelist[0](x)
    #             y1 = x
    #             for l in modulelist[1:]:
    #                 x = l(x)
    #             logits = x
    #             y00 = y0  # .clone().detach().requires_grad_(True)

    #             regu1, regu2 = df_dz_regularizer(odefunc, y00)
    #             regu1 = regu1.mean()
    #             regu2 = regu2.mean()
    #             # print("regu1:weight_diag " + str(regu1.item()) + ':' + str(weight_diag))
    #             # print("regu2:weight_offdiag " + str(regu2.item()) + ':' + str(weight_offdiag))
    #             regu3 = f_regularizer(odefunc, y00)
    #             regu3 = regu3.mean()
    #             # print("regu3:weight_f " + str(regu3.item()) + ':' + str(weight_f))
    #             loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2
    #             #         loss = weight_f*regu3

    #             if do_wandb:
    #                 wandb.log({
    #                     'phase2/test_reg1': regu1.item(),
    #                     'phase2/test_reg2': regu2.item(),
    #                     'phase2/test_reg3': regu3.item(),
    #                     'phase2/test_loss': loss.item(),
    #                 })


    if itr % batches_per_epoch == 0:
        if itr == 0:
            continue
        with torch.no_grad():
            if True:  # val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args},
                           os.path.join(odesavefolder, 'model_' + str(itr // batches_per_epoch) + '.pth'))

        
    # break # [X]
################################################ Phase 3, train final FC ################################################

endtime = 5
layernum = 0

# [X]
folder = f'./EXP/orth_dense_resnet_final_{exp_name}/model_39.pth'
saved = torch.load(folder)
print('load...', folder)
statedic = saved['state_dict']
args = saved['args']
tol = 1e-5
savefolder_fc = f'./EXP/orth_resnetfct5_15_cent_{exp_name}/'
print('saving...', savefolder_fc, ' endtime... ',endtime)


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x, t = None):
        self.integration_time = self.integration_time.type_as(x)
        if t is None: 
            out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        else: 
            integration_time = torch.tensor([0, t]).float().type_as(x)
            out = odeint(self.odefunc, x, integration_time, rtol=tol, atol=tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


makedirs(savefolder_fc)
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)]
fc_layers = [MLP_OUT()]
model = nn.Sequential(*feature_layers, *fc_layers).to(device)
model.load_state_dict(statedic) # [X]
for param in odefunc.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device)
regularizer = nn.MSELoss()

train_loader = DataLoader(DensemnistDatasetTrain(),
                          batch_size=128,
                          shuffle=True, num_workers=1
                          )
train_loader__ = DataLoader(DensemnistDatasetTrain(),
                            batch_size=128,
                            shuffle=True, num_workers=1
                            )
test_loader = DataLoader(DensemnistDatasetTest(),
                         batch_size=128,
                         shuffle=True, num_workers=1
                         )

data_gen = inf_generator(train_loader)
batches_per_epoch = len(train_loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

best_acc = 0
for itr in trange(5 * batches_per_epoch):

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)

    y = y.to(device)

    modulelist = list(model)

    y0 = x
    x = modulelist[0](x)
    y1 = x
    for l in modulelist[1:]:
        x = l(x)
    logits = x
    y00 = y0
    loss = criterion(logits, y)

    # # test reg1 reg2 reg3
    # regu1, regu2 = df_dz_regularizer(odefunc, y00)
    # regu1 = regu1.mean()
    # regu2 = regu2.mean()
    # # print("regu1:weight_diag " + str(regu1.item()) + ':' + str(weight_diag))
    # # print("regu2:weight_offdiag " + str(regu2.item()) + ':' + str(weight_offdiag))
    # regu3 = f_regularizer(odefunc, y00)
    # regu3 = regu3.mean()
    # # print("regu3:weight_f " + str(regu3.item()) + ':' + str(weight_f))
    # _loss = weight_f * regu3 + weight_diag * regu1 + weight_offdiag * regu2

    if do_wandb:
        wandb.log({
            'phase3/reg1': regu1.item(),
            'phase3/reg2': regu2.item(),
            'phase3/reg3': regu3.item(),
            # 'phase3/sodef_loss': _loss.item(),
            'phase3/ce_loss': loss.item(),
        })

    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    # break  # [X]

    if itr % batches_per_epoch == 0:
        if itr == 0:
            continue
        with torch.no_grad():
            val_acc = accuracy(model, test_loader)
            train_acc = accuracy(model, train_loader__)
            if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(savefolder_fc, 'model.pth'))
                best_acc = val_acc
            print(
                "Epoch {:04d}|Train Acc {:.4f} | Test Acc {:.4f}".format(
                    itr // batches_per_epoch, train_acc, val_acc
                )
            )

# Eval

net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

net = net.to(device)

net = nn.Sequential(*list(net.children())[0:-1])

fcs_temp = MLP_OUT_ORTH512()

fc_layers = MLP_OUT_BALL()
for param in fc_layers.parameters():
    param.requires_grad = False
net = nn.Sequential(*net, fcs_temp, fc_layers).to(device)

net.load_state_dict(torch.load(folder_savemodel + '/ckpt.pth')['net'])


statedict3 = torch.load(os.path.join(savefolder_fc, 'model.pth'))['state_dict']
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)]
fc_layers = [MLP_OUT()]
model2 = nn.Sequential(*feature_layers, *fc_layers).to(device)
model2.load_state_dict(statedict3) # [X]

print(model2)


import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()

        # Explicitly store layers from model A
        modules_a = list(model_a.children())

        self.backbone = nn.ModuleList(modules_a[:6])
        self.orth_head = modules_a[6]   # MLP_OUT_ORTH512_2

        # Model B layers
        modules_b = list(model_b.children())
        self.ode_block = modules_b[0]
        self.final_head = modules_b[1]

    def forward(self, x, return_feats=False, return_raw_logits=False):
        # CNN / ResNet part
        for layer in self.backbone:
            x = layer(x)

        # Spatial → vector collapse (CRUCIAL)
        x = x[..., 0, 0]      # shape: (B, 512)

        # Orthogonal head
        before_ode = self.orth_head(x) # shape: (B, 2)

        # ODE + classifier
        after_ode = self.ode_block(before_ode)
        x = self.final_head(after_ode)
        raw_logits = self.final_head(before_ode)

        if return_feats: 
            return before_ode, after_ode, x
        else:
            if return_raw_logits:
                return raw_logits, x
            else:
                return x 

combined_model = CombinedModel(net, model2)
combined_model


import torch
import torch.nn.functional as F
import numpy as np

def evaluate_standard(test_loader, model):
    denoised_test_loss = 0
    denoised_test_acc = 0
    raw_test_loss = 0
    raw_test_acc = 0
    n = 0

    # NEW: per-sample loss differences
    loss_diffs = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.cuda(), y.cuda()

            raw_output, denoised_output = model(X, return_raw_logits=True)

            # --- per-sample losses ---
            raw_loss_vec = F.cross_entropy(raw_output, y, reduction="none")
            denoised_loss_vec = F.cross_entropy(denoised_output, y, reduction="none")

            # accumulate difference: raw - denoised
            loss_diffs.append((raw_loss_vec - denoised_loss_vec).detach().cpu())

            # --- standard bookkeeping ---
            raw_test_loss += raw_loss_vec.sum().item()
            denoised_test_loss += denoised_loss_vec.sum().item()

            raw_test_acc += (raw_output.argmax(1) == y).sum().item()
            denoised_test_acc += (denoised_output.argmax(1) == y).sum().item()

            n += y.size(0)

    loss_diffs = torch.cat(loss_diffs, dim=0)  # shape (N,)

    return (
        raw_test_loss / n,
        raw_test_acc / n,
        denoised_test_loss / n,
        denoised_test_acc / n,
        loss_diffs
    )


upper_limit, lower_limit = 1, 0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False, no_ode_in_pgd_generation=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            raw_output, denoised_output = model((X + delta), return_raw_logits=True)
            if no_ode_in_pgd_generation: 
                output = raw_output
            else:
                output = denoised_output
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
        all_loss = F.cross_entropy(model((X + delta)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(
    test_loader,
    model,
    attack_iters=10,
    restarts=1,
    use_CWloss=False,
    no_ode_in_pgd_generation=False
):
    epsilon = 0.15
    alpha = 0.02

    raw_pgd_loss = 0
    raw_pgd_acc = 0
    denoised_pgd_loss = 0
    denoised_pgd_acc = 0
    n = 0

    # NEW: per-sample loss differences
    loss_diffs = []

    model.eval()
    for X, y in test_loader:
        X, y = X.cuda(), y.cuda()

        pgd_delta = attack_pgd(
            model, X, y,
            epsilon, alpha,
            attack_iters, restarts,
            use_CWloss=use_CWloss,
            no_ode_in_pgd_generation=no_ode_in_pgd_generation
        )

        with torch.no_grad():
            raw_output, denoised_output = model(X + pgd_delta, return_raw_logits=True)

            raw_loss_vec = F.cross_entropy(raw_output, y, reduction="none")
            denoised_loss_vec = F.cross_entropy(denoised_output, y, reduction="none")

            # accumulate difference
            loss_diffs.append((raw_loss_vec - denoised_loss_vec).detach().cpu())

            raw_pgd_loss += raw_loss_vec.sum().item()
            denoised_pgd_loss += denoised_loss_vec.sum().item()

            raw_pgd_acc += (raw_output.argmax(1) == y).sum().item()
            denoised_pgd_acc += (denoised_output.argmax(1) == y).sum().item()

            n += y.size(0)

    loss_diffs = torch.cat(loss_diffs, dim=0)

    return (
        raw_pgd_loss / n,
        raw_pgd_acc / n,
        denoised_pgd_loss / n,
        denoised_pgd_acc / n,
        loss_diffs
    )

# Clean acc 
raw_test_loss, raw_test_acc, denoised_test_loss, denoised_test_acc, loss_diffs_clean = evaluate_standard(testloader, combined_model)
print("raw_test_loss", raw_test_loss)
print("raw_test_acc", raw_test_acc)
print("denoised_test_loss", denoised_test_loss)
print("denoised_test_acc", denoised_test_acc)


import numpy as np
import matplotlib.pyplot as plt

def plot_loss_diff_histogram_trimmed(
    loss_diffs,
    save_path,
    title="Loss Difference Histogram (Raw − Denoised)",
    max_bins=200,
    trim_q=0.01   # trim 1% on each side
):
    # Convert to numpy
    if hasattr(loss_diffs, "numpy"):
        loss_diffs = loss_diffs.numpy()

    loss_diffs = loss_diffs.flatten()

    # Quantile-based trimming
    low_q = np.quantile(loss_diffs, trim_q)
    high_q = np.quantile(loss_diffs, 1 - trim_q)

    trimmed = loss_diffs[(loss_diffs >= low_q) & (loss_diffs <= high_q)]

    # Freedman–Diaconis binning on trimmed data
    q25, q75 = np.percentile(trimmed, [25, 75])
    iqr = q75 - q25

    min_val = trimmed.min()
    max_val = trimmed.max()

    if iqr > 0:
        bin_width = 2 * iqr * (len(trimmed) ** (-1 / 3))
        bins = int(np.clip((max_val - min_val) / bin_width, 20, max_bins))
    else:
        bins = min(50, max_bins)

    plt.figure(figsize=(6.5, 4.8))

    plt.hist(
        trimmed,
        bins=bins,
        range=(min_val, max_val),
        density=True,
        alpha=0.85
    )

    # Reference line
    plt.axvline(0.0, linestyle="--", linewidth=2)

    plt.xlim(min_val, max_val)
    plt.xlabel("Loss difference (raw − denoised)")
    plt.ylabel("Density")
    plt.title(title + f"\n(trimmed {int(trim_q*100)}% tails)")

    # Stats (reported on FULL data, not trimmed)
    mean_full = loss_diffs.mean()
    median_full = np.median(loss_diffs)
    pct_helped = (loss_diffs > 0).mean() * 100

    plt.text(
        0.02, 0.96,
        f"Mean Δℓ (full)   = {mean_full:.4f}\n"
        f"Median Δℓ (full) = {median_full:.4f}\n"
        f"% helped         = {pct_helped:.1f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10
    )

    plt.tight_layout()
    if do_wandb: 
        # wandb.log({f"{save_path}": plt})
        wandb.log({f"{exp_name} {save_path}": wandb.Image(plt)})

    plt.savefig(save_path, dpi=300)
    plt.clf()
    plt.close()

plot_loss_diff_histogram_trimmed(
    loss_diffs_clean,
    save_path="loss_diff_clean.png",
    title="Clean Data: Raw − Denoised Loss"
)

# PGD generation w/ODE
no_ode_raw_test_loss, no_ode_raw_test_acc, no_ode_denoised_test_loss, no_ode_denoised_test_acc, loss_diffs_pgd_w_ode = evaluate_pgd(testloader, combined_model, no_ode_in_pgd_generation=False)

print("no_ode_raw_test_loss", no_ode_raw_test_loss)
print("no_ode_raw_test_acc", no_ode_raw_test_acc)
print("no_ode_denoised_test_loss", no_ode_denoised_test_loss)
print("no_ode_denoised_test_acc", no_ode_denoised_test_acc)

plot_loss_diff_histogram_trimmed(
    loss_diffs_pgd_w_ode,
    save_path="loss_diff_pgd_w_ode.png",
    title="Adv Data (pgd w/ode): Raw − Denoised Loss"
)

# PGD generation w.o./ODE
w_ode_raw_test_loss, w_ode_raw_test_acc, w_ode_denoised_test_loss, w_ode_denoised_test_acc, loss_diffs_pgd_w_o_ode = evaluate_pgd(testloader, combined_model, no_ode_in_pgd_generation=True)

print("w_ode_raw_test_loss", w_ode_raw_test_loss)
print("w_ode_raw_test_acc", w_ode_raw_test_acc,)
print("w_ode_denoised_test_loss", w_ode_denoised_test_loss,)
print("w_ode_denoised_test_acc", w_ode_denoised_test_acc,)



plot_loss_diff_histogram_trimmed(
    loss_diffs_pgd_w_o_ode,
    save_path="loss_diff_pgd_w_o_ode.png",
    title="Adv Data (pgd w.o./ode): Raw − Denoised Loss"
)

@torch.no_grad()
def estimate_basin_stats_relative(
    model,
    stable_points,
    T_test=5.0,
    alphas=(0.001, 0.01, ), # , 0.03, 0.05, 0.1   # relative noise levels
    samples_per_alpha=10,
    beta=0.01,                        # relative tolerance (1%)
    eps=1e-8,
    use_settle_test=False
):
    """
    Returns: dict[class][point_idx][alpha] -> success rate

    success definition (default):
      || z(T) - z* || <= beta * max(||z*||, eps)

    If use_settle_test=True:
      success if || z2 - z1 || <= beta * max(||z1||, eps)
      where z1 = flow(z0), z2 = flow(z1)
    """
    basin_stats = {}

    for c, points in stable_points.items():
        class_stats = []
        for z_star in points:
            z_star = z_star.detach()
            norm_star = torch.norm(z_star).clamp(min=eps)

            alpha_stats = {}

            for alpha in alphas:
                success = 0
                for _ in range(samples_per_alpha):
                    # relative perturbation
                    delta = alpha * torch.randn_like(z_star) * norm_star
                    z0 = (z_star + delta).unsqueeze(0)

                    z1 = model.ode_block(z0, t=T_test).squeeze(0)
                    print(torch.norm(z1 - z_star))
                    if not use_settle_test:
                        thresh = beta * norm_star
                        ok = (torch.norm(z1 - z_star) <= thresh)
                    else:
                        z2 = model.ode_block(z1.unsqueeze(0), t=T_test).squeeze(0)
                        thresh = beta * torch.norm(z1).clamp(min=eps)
                        ok = (torch.norm(z2 - z1) <= thresh)

                    success += int(ok)

                alpha_stats[alpha] = success / samples_per_alpha

            class_stats.append(alpha_stats)
        basin_stats[c] = class_stats

    return basin_stats

def collect_feats(test_loader, model):
    model.eval()

    feats_before_all = []
    feats_after_all = []
    labels_all = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.cuda()
            y = y.cuda()

            feats_before, feats_after, output = model(X, return_feats=True)

            feats_before_all.append(feats_before)
            feats_after_all.append(feats_after)
            labels_all.append(y)

    feats_before_all = torch.cat(feats_before_all, dim=0)  # (N, 2)
    feats_after_all  = torch.cat(feats_after_all, dim=0)   # (N, 2)
    labels_all       = torch.cat(labels_all, dim=0)        # (N,)

    return feats_before_all, feats_after_all, labels_all

feats_before, feats_after, labels = collect_feats(testloader, combined_model)

@torch.no_grad()
def integrate_to_long_T(model, z0, T_long):
    """
    z0: (N, d)
    """
    return model.ode_block(z0, t=T_long)
T_long = 20.0
zT = integrate_to_long_T(combined_model, feats_before, T_long)

from sklearn.cluster import KMeans

def extract_stable_points_per_class(
    zT,
    labels,
    num_classes,
    clusters_per_class=1
):
    """
    returns: dict[class] -> tensor (k, d)
    """
    centers = {}

    zT_np = zT.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for c in range(num_classes):
        zc = zT_np[labels_np == c]
        if len(zc) == 0:
            continue

        kmeans = KMeans(
            n_clusters=clusters_per_class,
            n_init=10,
            random_state=0
        ).fit(zc)

        centers[c] = torch.tensor(
            kmeans.cluster_centers_,
            device=zT.device,
            dtype=zT.dtype
        )

    return centers

num_classes = 10
stable_points = extract_stable_points_per_class(
    zT, labels, num_classes
)


print(estimate_basin_stats_relative(combined_model, stable_points))