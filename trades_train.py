import argparse
import copy
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
from autoattack import AutoAttack
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from torch.utils.data import Dataset, DataLoader

# from temp_util import progress_bar 
# from model import *
# from utils import *

# from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
#     attack_pgd, evaluate_pgd, evaluate_standard, normalize)

from _utils import (
    get_loaders, Identity, MLP_OUT_ORTH1024, MLP_OUT_BALL, inf_generator, progress_bar, save_training_feature, save_testing_feature, makedirs, ODEfunc_mlp, MLP_OUT_LINEAR, MLP_OUT_ORTH512, accuracy
)
from torchdiffeq import odeint_adjoint as odeint

def phase_one(trainloader, testloader, train_eval_loader, max_row_dis_64_10_path, device): 
    """
    Phase 1, save robust feature from backbone
    """
    nepochs_save_robustfeature = 4
    robust_feature_savefolder = './EXP/CIFAR10_resnet_Nov_1'

    # Load the pretrained model 
    from torchinfo import summary
    from robustbench import load_model
    # trades_backbone = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
    ###  trades_backbone = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    from resnet import ResNet18
    trades_backbone = ResNet18()
    summary(trades_backbone, (1, 3, 32, 32))
    print(trades_backbone)

    # Create an orthogonal bridge layer: 
    # trades_backbone.logits = Identity()
    trades_backbone.linear = Identity()
    orthogonal_bridge_layer = MLP_OUT_ORTH512()
    # orthogonal_bridge_layer = MLP_OUT_ORTH1024()
    
    # create a max row dist. FC layer for classification, this is only used here to guide the bridge layer 
    fc_layer_phase1 = MLP_OUT_BALL(max_row_dis_64_10_path)

    # The backbone grad is off
    # The bridge layer is On
    # The fc layer is off
    for param in fc_layer_phase1.parameters():
        param.requires_grad = False
    for param in trades_backbone.parameters():
        param.requires_grad = False
    phase1_model = nn.Sequential(trades_backbone, orthogonal_bridge_layer, fc_layer_phase1).to(device)

     
    print(phase1_model)
    phase1_model = phase1_model.to(device)
    
    
    data_gen = inf_generator(trainloader)
    batches_per_epoch = len(trainloader)


    best_acc = 0  # best test accuracy
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(phase1_model.parameters(), lr=1e-1, eps=1e-2, amsgrad=True)

    makedirs(robust_feature_savefolder)
    from _utils import train_save_robustfeature, test_save_robustfeature

    for epoch in range(0, nepochs_save_robustfeature):
        train_save_robustfeature(epoch, phase1_model, trainloader, device, optimizer, criterion)
        # break
        best_acc = test_save_robustfeature(epoch, phase1_model, testloader, device, criterion, best_acc, robust_feature_savefolder, train_eval_loader)
        print('save robust feature to ' + robust_feature_savefolder)
    
    saved_temp = torch.load(robust_feature_savefolder+'/ckpt.pth')
    statedic_temp = saved_temp['net_save_robustfeature']
    phase1_model.load_state_dict(statedic_temp)
        
    return trades_backbone, orthogonal_bridge_layer

# device = torch.device('cuda:0') 
device = 'cpu'


robust_feature_savefolder = './EXP/CIFAR10_resnet_Nov_1'
train_savepath='./data/CIFAR10_train_resnetNov1.npz'
test_savepath='./data/CIFAR10_test_resnetNov1.npz'
    
ODE_FC_save_folder = robust_feature_savefolder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


args = get_args()
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


trainloader, testloader, train_eval_loader, _ = get_loaders(args.data_dir, args.batch_size)
robust_backbone, robust_backbone_fc_features = phase_one(trainloader, testloader, train_eval_loader, 'fc_max_row_64_10.pth', device)

################################################ Phase 2, train ODE block ################################################    
def phase2(ODE_FC_save_folder, train_savepath, test_savepath):
    weight_diag = 10
    weight_offdiag = 0
    weight_f = 0.1

    weight_norm = 0
    weight_lossc =  0

    exponent = 1.0
    exponent_off = 0.1 
    exponent_f = 50
    time_df = 1
    trans = 1.0
    transoffdig = 1.0
    numm = 16


    ODE_FC_odebatch = 32
    ODE_FC_ode_epoch = 20
    from _utils import ODEBlocktemp, df_dz_regularizer, f_regularizer
    from _utils import temp1, temp2
    from _utils import DensemnistDatasetTest, DensemnistDatasetTrain

    makedirs(ODE_FC_save_folder)

    
    odefunc = ODEfunc_mlp(0)
    feature_layers = ODEBlocktemp(odefunc)
    fc_layers = MLP_OUT_LINEAR()
    for param in fc_layers.parameters():
        param.requires_grad = False

    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)

    train_loader_ODE =  DataLoader(DensemnistDatasetTrain(train_savepath),
        batch_size=ODE_FC_odebatch,
        shuffle=True, num_workers=2
    )
    train_loader_ODE__ =  DataLoader(DensemnistDatasetTrain(train_savepath),
        batch_size=ODE_FC_odebatch,
        shuffle=True, num_workers=2
    )

    test_loader_ODE =  DataLoader(DensemnistDatasetTest(test_savepath),
        batch_size=ODE_FC_odebatch,
        shuffle=True, num_workers=2
    )

    data_gen = inf_generator(train_loader_ODE)
    batches_per_epoch = len(train_loader_ODE)

    optimizer = torch.optim.Adam(ODE_FCmodel.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)

    for itr in range(ODE_FC_ode_epoch * batches_per_epoch):

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)

        modulelist = list(ODE_FCmodel)
        y0 = x
        x = modulelist[0](x)
        y1 = x

    #         y00 = y1.clone().detach().requires_grad_(True)
        y00 = y0#.clone().detach().requires_grad_(True)

        regu1, regu2  = df_dz_regularizer(_, y00, numm=numm, odefunc=odefunc, time_df=time_df, exponent=exponent, trans=trans, exponent_off=exponent_off, transoffdig=transoffdig, device=device)
        regu1 = regu1.mean()
        regu2 = regu2.mean()
        print("regu1:weight_diag "+str(regu1.item())+':'+str(weight_diag))
        print("regu2:weight_offdiag "+str(regu2.item())+':'+str(weight_offdiag))
        regu3 = f_regularizer(_, y00, odefunc=odefunc, time_df=time_df, device=device, exponent_f=exponent_f)
        regu3 = regu3.mean()
        print("regu3:weight_f "+str(regu3.item())+':'+str(weight_f))
        loss = weight_f*regu3 + weight_diag*regu1+ weight_offdiag*regu2


        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        if itr % batches_per_epoch == 0:
            if itr ==0:
                recordtext = os.path.join(ODE_FC_save_folder, 'output.txt')
                if os.path.isfile(recordtext):
                    os.remove(recordtext)
                # continue

            with torch.no_grad():
                if True:#val_acc > best_acc:
                    torch.save({'state_dict': ODE_FCmodel.state_dict()}, os.path.join(ODE_FC_save_folder, 'model_'+str(itr // batches_per_epoch)+'.pth'))
                with open(recordtext, "a") as text_file:
                    text_file.write("Epoch {:04d}".format(itr // batches_per_epoch)+'\n')

                    temp1(odefunc, y00, text_file, numm=numm, odefunc=odefunc, time_df=time_df, device=device, exponent=exponent, trans=trans, exponent_off=exponent_off, transoffdig=transoffdig)
                    temp2(odefunc, y00, text_file, odefunc=odefunc, time_df=time_df, device=device, exponent_f=exponent_f)

                    text_file.close()
        # break

    return ODE_FCmodel, odefunc

ODE_FCmodel, odefunc = phase2(ODE_FC_save_folder = robust_feature_savefolder, train_savepath=train_savepath, test_savepath=test_savepath)


################################################ Phase 3, train final FC ################################################    
def phase3(odefunc, robust_backbone, robust_backbone_fc_features, trainloader, testloader):
    from _utils import ODEBlock

    ODE_FC_fcbatch = 128
    ODE_FC_fc_epoch = 10

    feature_layers = ODEBlock(odefunc)
    fc_layers = MLP_OUT_LINEAR()
    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)
    
    # saved = torch.load('./EXP/CIFAR10_resnetNov/dense1_temp/model_.pth')
    # statedic = saved['state_dict']
    # ODE_FCmodel.load_state_dict(statedic)

    for param in odefunc.parameters():
        param.requires_grad = True
    for param in robust_backbone_fc_features.parameters():
        param.requires_grad = False
    for param in robust_backbone.parameters():
        param.requires_grad = False

    new_model_full = nn.Sequential(robust_backbone, robust_backbone_fc_features, ODE_FCmodel).to(device)

    
    optimizer = torch.optim.Adam([{'params': odefunc.parameters(), 'lr': 1e-5, 'eps':1e-6,},
                                {'params': fc_layers.parameters(), 'lr': 1e-2, 'eps':1e-4,}], amsgrad=True)

    criterion = nn.CrossEntropyLoss()

    from _utils import train_phase3, test_phase3
    
    best_acc = 0
        
    for epoch in range(0, ODE_FC_fc_epoch):
        
        train_phase3(new_model_full, epoch, trainloader=trainloader, optimizer=optimizer, criterion=criterion, device=device)
        # train_mixup(new_model_full, epoch)

        with torch.no_grad():
            val_acc = accuracy(new_model_full, testloader, device)
            if val_acc > best_acc:
                torch.save({'state_dict': new_model_full.state_dict()}, os.path.join(ODE_FC_save_folder, 'full.pth'))
                best_acc = val_acc
            print("Epoch {:04d} |  Test Acc {:.4f}".format(epoch,  val_acc))
            
        # break

    # saved_temp = torch.load(os.path.join(ODE_FC_save_folder, 'full.pth'))
    # statedic_temp = saved_temp['state_dict']
    # new_model_full.load_state_dict(statedic_temp)
    return new_model_full

new_model_full = phase3(odefunc, robust_backbone, robust_backbone_fc_features, trainloader=trainloader, testloader=testloader)

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model((X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            break
    return test_loss/n, test_acc/n

print('CLEAN, Test Loss, Test Acc', evaluate_standard(testloader, new_model_full))

# ################################################ attack ################################################   

# l = [x for (x, y) in testloader]
# x_test = torch.cat(l, 0)
# l = [y for (x, y) in testloader]
# y_test = torch.cat(l, 0)


# ##### here we split the set to multi servers and gpus to speed up the test. otherwise it is too slow.
# ##### if your server is powerful or your have enough time, just use the full dataset directly by commenting out the following.
# #############################################    
# iii = 9

# x_test = x_test[1000*iii:1000*(iii+1),...]
# y_test = y_test[1000*iii:1000*(iii+1),...]

# #############################################   

# print('run_standard_evaluation_individual', 'Linf')
# print(x_test.shape)


# epsilon = 8 / 255.
# adversary = AutoAttack(new_model_full, norm='Linf', eps=epsilon, version='standard')




# X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
# torch.save({'state_dict': new_model_full.state_dict()}, os.path.join(ODE_FC_save_folder, 'full.pth'))