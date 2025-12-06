import numpy as np 
import torch 
import os 
from _utils import get_loaders
from torchinfo import summary
import torch.nn as nn 

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

SEED = 11
# device = 'cpu'
device = torch.device('cuda:0') 
BATCH_SIZE = 128
CIFAR10_DIR = './data'
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.enabled = True  # Enables cudnn
    torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.deterministic = True  # To have ~deterministic results


def load_clean_model(path: str = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/pytorch-cifar/checkpoint/ckpt.pth'): 
    print('loading clean model')
    from resnet import ResNet18
    model = ResNet18()  
    k1, k2 = model.load_state_dict(torch.load(path)['net'])
    print(k1, k2)

    model.float()
    model.eval()

    return model 

def load_sodef_model(path: str = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/EXP/CIFAR10_resnet_Nov_1/full.pth'): 
    print('loading sodef model')
    from resnet import ResNet18
    from _utils import MLP_OUT_ORTH512, MLP_OUT_LINEAR, ODEBlock, ODEfunc_mlp
    backbone = ResNet18() 
    orthogonal_bridge = MLP_OUT_ORTH512()

    odefunc = ODEfunc_mlp(0)
    feature_layers = ODEBlock(odefunc)
    fc_layers = MLP_OUT_LINEAR()
    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)
    
    sodef_model = nn.Sequential(backbone, orthogonal_bridge, ODE_FCmodel)
    print(sodef_model)
    
    k1, k2 = sodef_model.load_state_dict(torch.load(path)['state_dict'])
    print(k1, k2)


train_loader, test_loader, _, test_set = get_loaders(CIFAR10_DIR, BATCH_SIZE, "CIFAR10")

    
clean_model = load_clean_model().to(device)
sodef_model = load_sodef_model().to(device)

from _utils import evaluate_standard, evaluate_pgd
print('CLEAN MODEL, Test Loss, Test Acc', evaluate_standard(test_loader, clean_model, device))
print('SODEF MODEL, Test Loss, Test Acc', evaluate_standard(test_loader, sodef_model, device))

print('CLEAN MODEL | PGD 8/255, Test Loss, Test Acc', evaluate_pgd(test_loader, clean_model, attack_iters=20, restarts=1, eps=8, step=1, use_CWloss=False))
print('SODEF MODEL, Test Loss, Test Acc', evaluate_pgd(test_loader, sodef_model, attack_iters=20, restarts=1, eps=8, step=1, use_CWloss=False))


# !wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
# !tar -xf /content/SODEF/trades_r/CIFAR-10-C.tar

