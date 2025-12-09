# Build dataset from saved features 
# make the orth + max_dist layer
# train w/ CE 
import torch
import torch.nn as nn 
import numpy as np 
from tqdm import trange
from torch.utils.data import DataLoader
import sys 


from utils_bert import get_sst2_feature_dataset, get_bert_fc_layer, Logger
from _utils import makedirs
from _utils import MLP_OUT_ORTH_X_X, MLP_OUT_BALL_given_mat
from utils_bert import get_max_row_dist_for_2_classes, check_max_row_dist_matrix, train_ce, test_ce

LOG_PATH = 'testingBertSodef'
EXP_NAME = 'onos'

makedirs(f'{LOG_PATH}/{EXP_NAME}')
sys.stdout = Logger("{}/{}/run.log".format(LOG_PATH, EXP_NAME))
# sys.stderr = sys.stdout
command_line_args = sys.argv
command = " ".join(command_line_args)
print(f"The command that ran this script: {command}")

BERT_CKPT_DIR = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2'
FEATS_DIR = f'{BERT_CKPT_DIR}/saving_feats/{0}_feats.npz'
CLF_LAYER_DIR = f'{BERT_CKPT_DIR}/bert_clf.pth'
PHASE1_BS = 128

device = 'cpu' if not torch.cuda.is_available() else torch.device('cuda:0')

def bert_fc_features_sanity_check(bert_clf_layer, trainloader, testloader, device): 
    tr_res = test_ce(-1, bert_clf_layer, trainloader, device, nn.CrossEntropyLoss(), 110, '')
    te_res = test_ce(-1, bert_clf_layer, testloader, device, nn.CrossEntropyLoss(), 110, '')

    print('Train Acc, Loss', tr_res['acc'], tr_res['loss'])
    print('Test Acc, Loss', te_res['acc'], te_res['loss'])

def phase1(trainloader, testloader, device, load_phase1: bool = False, base_folder: str = None): 
    feature_dim = 768
    n_epochs = 4 
    bridge_dim = 64
    phase1_save_folder = f'{base_folder}/phase1'
    makedirs(phase1_save_folder)

    orthogonal_bridge_layer = MLP_OUT_ORTH_X_X(feature_dim, bridge_dim) # make it so you pass bridge_dim
    max_row_mat = get_max_row_dist_for_2_classes(bridge_dim)
    check_max_row_dist_matrix(max_row_mat.cpu().numpy().T, 2)
    fc_layer_phase1 = MLP_OUT_BALL_given_mat(max_row_mat, dim=bridge_dim, num_classes=2)
    
    phase1_model = nn.Sequential(orthogonal_bridge_layer, fc_layer_phase1).to(device)
    print('phase1_model', phase1_model)
    
    ### 
    best_acc = 0  # best test accuracy
    criterion = nn.CrossEntropyLoss() ### 
    optimizer = torch.optim.Adam(phase1_model.parameters(), lr=1e-1, eps=1e-2, amsgrad=True)

    if not load_phase1: 
        for epoch in trange(0, n_epochs):
            tr_results = train_ce(epoch, phase1_model, trainloader, device, optimizer, criterion)
            print('tr_acc, tr_loss', tr_results['acc'], tr_results['loss'])
            te_results = test_ce(epoch, phase1_model, testloader, device, criterion, best_acc, phase1_save_folder)
            best_acc = te_results['best_acc']
            print('te_acc, te_loss', te_results['acc'], te_results['loss'])
            
    else:     
        saved_temp = torch.load(phase1_save_folder+'/phase1_best_acc_ckpt.pth')
        statedic_temp = saved_temp['phase1_model']
        phase1_model.load_state_dict(statedic_temp)
            
    return orthogonal_bridge_layer


sst2_train_feature_set, sst2_test_feature_set = get_sst2_feature_dataset(FEATS_DIR)
train_feature_loader = DataLoader(sst2_train_feature_set,
    batch_size=PHASE1_BS,
    shuffle=True, num_workers=2
)
test_feature_loader = DataLoader(sst2_test_feature_set,
    batch_size=PHASE1_BS,
    shuffle=False, num_workers=2
)

bert_fc_layer = get_bert_fc_layer(CLF_LAYER_DIR).to(device)

bert_fc_features_sanity_check(bert_fc_layer, train_feature_loader, test_feature_loader, device)

orth_bridge_layer = phase1(train_feature_loader, test_feature_loader, device, False, f'{LOG_PATH}/{EXP_NAME}')
