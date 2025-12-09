import numpy as np 
import torch 
import os 
from _utils import get_loaders
from torchinfo import summary
import torch.nn as nn 

import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm

from torchvision import transforms
import os

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
    from _utils import MLP_OUT_ORTH512, MLP_OUT_LINEAR, ODEBlock, ODEfunc_mlp, Identity
    backbone = ResNet18() 
    backbone.linear = Identity()
    orthogonal_bridge = MLP_OUT_ORTH512()

    odefunc = ODEfunc_mlp(0)
    feature_layers = ODEBlock(odefunc)
    fc_layers = MLP_OUT_LINEAR()
    ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)
    
    sodef_model = nn.Sequential(backbone, orthogonal_bridge, ODE_FCmodel)
    k1, k2 = sodef_model.load_state_dict(torch.load(path)['state_dict'])
    print(k1, k2)

    return sodef_model

train_loader, test_loader, _, test_set = get_loaders(CIFAR10_DIR, BATCH_SIZE, "CIFAR10")

    
clean_model = load_clean_model().to(device)
sodef_model = load_sodef_model().to(device)

from _utils import evaluate_standard, evaluate_pgd
print('CLEAN MODEL, Test Loss, Test Acc', evaluate_standard(test_loader, clean_model, device))
print('SODEF MODEL, Test Loss, Test Acc', evaluate_standard(test_loader, sodef_model, device))

# print('CLEAN MODEL | PGD 8/255, Test Loss, Test Acc', evaluate_pgd(test_loader, clean_model, attack_iters=20, restarts=1, eps=8, step=1, use_CWloss=False))
# print('SODEF MODEL, Test Loss, Test Acc', evaluate_pgd(test_loader, sodef_model, attack_iters=20, restarts=1, eps=8, step=1, use_CWloss=False))

"""
CLEAN MODEL, Test Loss, Test Acc (0.18003013288974762, 0.9538)
SODEF MODEL, Test Loss, Test Acc (0.25109421396255494, 0.9431)
CLEAN MODEL | PGD 8/255, Test Loss, Test Acc (4.279358312988281, 0.0004)
SODEF MODEL, Test Loss, Test Acc (2.214425745010376, 0.3433)
"""

# !wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
# !tar -xf /content/SODEF/trades_r/CIFAR-10-C.tar

from _utils import CIFAR10_C, eval_cifar10c_name

def get_cifar10c_acc(model, data_root: str, device): 
    model = model.to(device)
    model.eval()

    # mean_accs = eval_cifar10c_severity(
    #     root=TEST_DATA_ROOT, model=model, device=DEVICE,)

    # for i in range(len(mean_accs)):
    #     print(f'cifar10c-severity-{i+1}-acc', (mean_accs[i]))

    corruptes = ['brightness', 'elastic_transform', 'gaussian_blur', 'impulse_noise',
        'motion_blur', 'shot_noise', 'speckle_noise', 'contrast', 'fog', 'gaussian_noise',
        'jpeg_compression', 'pixelate', 'snow', 'zoom_blur', 'defocus_blur', 'frost', 'glass_blur',
        'saturate', 'spatter']

    accs = []
    for name in corruptes:
        cifar10c_acc = eval_cifar10c_name(
            root=data_root, model=model, device=device, name=name,
            batch_size=BATCH_SIZE, )
        accs.append(cifar10c_acc)
        print(f'test (cifar10c-{name}) accuracy: {cifar10c_acc}')
    print('Cifar10-C', np.array(accs).mean())

    return accs

data_root = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/datasets/cifar10c'
print('getting clean model accs')
clean_model_accs = get_cifar10c_acc(clean_model, data_root, device)
print('getting sodef model accs')
sodef_model_accs = get_cifar10c_acc(sodef_model, data_root, device)

"""
CLEAN MODEL: 
test (cifar10c-brightness) accuracy: 0.94112
test (cifar10c-elastic_transform) accuracy: 0.8565
test (cifar10c-gaussian_blur) accuracy: 0.73862
test (cifar10c-impulse_noise) accuracy: 0.53148
test (cifar10c-motion_blur) accuracy: 0.80182
test (cifar10c-shot_noise) accuracy: 0.60228
test (cifar10c-speckle_noise) accuracy: 0.64242
test (cifar10c-contrast) accuracy: 0.79884
test (cifar10c-fog) accuracy: 0.89902
test (cifar10c-gaussian_noise) accuracy: 0.4705
test (cifar10c-jpeg_compression) accuracy: 0.8027
test (cifar10c-pixelate) accuracy: 0.77314
test (cifar10c-snow) accuracy: 0.84446
test (cifar10c-zoom_blur) accuracy: 0.80068
test (cifar10c-defocus_blur) accuracy: 0.83738
test (cifar10c-frost) accuracy: 0.80884
test (cifar10c-glass_blur) accuracy: 0.56118
test (cifar10c-saturate) accuracy: 0.9261
test (cifar10c-spatter) accuracy: 0.85468
Cifar10-C 0.762724210526315
SODEF MODEL: 
test (cifar10c-brightness) accuracy: 0.94176
test (cifar10c-elastic_transform) accuracy: 0.83848
test (cifar10c-gaussian_blur) accuracy: 0.73286
test (cifar10c-impulse_noise) accuracy: 0.49838
test (cifar10c-motion_blur) accuracy: 0.79482
test (cifar10c-shot_noise) accuracy: 0.55988
test (cifar10c-speckle_noise) accuracy: 0.59226
test (cifar10c-contrast) accuracy: 0.83328
test (cifar10c-fog) accuracy: 0.90236
test (cifar10c-gaussian_noise) accuracy: 0.44582
test (cifar10c-jpeg_compression) accuracy: 0.76416
test (cifar10c-pixelate) accuracy: 0.73912
test (cifar10c-snow) accuracy: 0.84294
test (cifar10c-zoom_blur) accuracy: 0.78788
test (cifar10c-defocus_blur) accuracy: 0.82904
test (cifar10c-frost) accuracy: 0.82334
test (cifar10c-glass_blur) accuracy: 0.55078
test (cifar10c-saturate) accuracy: 0.90094
test (cifar10c-spatter) accuracy: 0.828
Cifar10-C 0.7476894736842105
"""