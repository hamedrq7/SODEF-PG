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
from utils_bert import get_max_row_dist_for_2_classes, check_max_row_dist_matrix, train_ce, test_ce, set_seed_reproducability

LOG_PATH = 'testingBertSodef'
EXP_NAME = 'duos'
SEED = 42

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
PHASE2_BS = 32 

device = 'cpu' if not torch.cuda.is_available() else torch.device('cuda:0')
set_seed_reproducability(SEED)

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
            
    return phase1_model


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

phase1_model = phase1(train_feature_loader, test_feature_loader, device, base_folder = f'{LOG_PATH}/{EXP_NAME}',
                      load_phase1=True
                      )

class Phase2Model(nn.Module): 
    def __init__(self, bridge_768_64, ode_block, fc): 
        super(Phase2Model, self).__init__()
        self.bridge_768_64 = bridge_768_64
        self.ode_block = ode_block
        self.fc = fc

    def freeze_layer_given_name(self, layer_names): 
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for target in layer_names:
            if target in self._modules:   # only top-level modules
                module = self._modules[target]
                for param in module.parameters():
                    param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad == True: 
                print(f"[TRAINABLE] {name}")
            else:
                print(f"[FROZEN] {name}")
    
    def forward(self, x): 
        before_ode_feats = self.bridge_768_64(x)
        after_ode_feats = self.ode_block(before_ode_feats)
        logits = self.fc(after_ode_feats)
        return before_ode_feats, after_ode_feats, logits
    
class SingleOutputWrapper(nn.Module): 
    def __init__(self, model): 
        super(SingleOutputWrapper, self).__init__()
        self.model = model 

    def forward(self, x): 
        _, _, out = self.model(x)
        return out 
    
def phase2(bridge_768_64, trainloader, testloader, ODE_FC_save_folder, load_phase2_path: str = None, fc_layer = None):
    import os 
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
    from _utils import ODEfunc_mlp, MLP_OUT_LINEAR, inf_generator

    makedirs(ODE_FC_save_folder)

    odefunc = ODEfunc_mlp(0)
    use_fc_from_phase1 = not fc_layer is None
    print('use_fc_from_phase1', use_fc_from_phase1)
    phase2_model = Phase2Model(bridge_768_64, ODEBlocktemp(odefunc), MLP_OUT_LINEAR(dim1=64, dim2=2) if not use_fc_from_phase1 else fc_layer)
    phase2_model.freeze_layer_given_name(['bridge_768_64', 'fc'])
    phase2_model = phase2_model.to(device)

    train_data_gen = inf_generator(trainloader)
    batches_per_epoch = len(trainloader)

    optimizer = torch.optim.Adam(phase2_model.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
    print(ODE_FC_ode_epoch * batches_per_epoch)

    if load_phase2_path is None: 
        for itr in range(ODE_FC_ode_epoch * batches_per_epoch):

            optimizer.zero_grad()
            x, y = train_data_gen.__next__()
            x = x.to(device)

            # modulelist = list(ODE_FCmodel)
            # y0 = x
            # x = modulelist[0](x)
            # y1 = x
            # y00 = y0 #.clone().detach().requires_grad_(True)
            x, y00, _ = phase2_model(x)
            regu1, regu2  = df_dz_regularizer(None, x, numm=numm, odefunc=odefunc, time_df=time_df, exponent=exponent, trans=trans, exponent_off=exponent_off, transoffdig=transoffdig, device=device)
            regu1 = regu1.mean()
            regu2 = regu2.mean()
            print("regu1:weight_diag "+str(regu1.item())+':'+str(weight_diag))
            print("regu2:weight_offdiag "+str(regu2.item())+':'+str(weight_offdiag))
            regu3 = f_regularizer(None, x, odefunc=odefunc, time_df=time_df, device=device, exponent_f=exponent_f)
            regu3 = regu3.mean()
            print("regu3:weight_f "+str(regu3.item())+':'+str(weight_f))
            loss = weight_f*regu3 + weight_diag*regu1+ weight_offdiag*regu2

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            if itr % batches_per_epoch == 0:
                tr_res = test_ce(-1, SingleOutputWrapper(phase2_model), trainloader, device, nn.CrossEntropyLoss(), 110, '')
                te_res = test_ce(-1, SingleOutputWrapper(phase2_model), testloader, device, nn.CrossEntropyLoss(), 110, '')
                print('itr = ', itr, 'Train Acc, Loss', tr_res['acc'], tr_res['loss'])
                print('itr = ', itr, 'Test Acc, Loss', te_res['acc'], te_res['loss'])
                if itr ==0:
                    recordtext = os.path.join(ODE_FC_save_folder, 'output.txt')
                    if os.path.isfile(recordtext):
                        os.remove(recordtext)
                    # continue

                with torch.no_grad():
                    if True:#val_acc > best_acc:
                        torch.save({'state_dict': phase2_model.state_dict()}, os.path.join(ODE_FC_save_folder, 'phase2model_'+str(itr // batches_per_epoch)+'.pth'))
                    with open(recordtext, "a") as text_file:
                        text_file.write("Epoch {:04d}".format(itr // batches_per_epoch)+'\n')

                        temp1(odefunc, y00, text_file, numm=numm, odefunc=odefunc, time_df=time_df, device=device, exponent=exponent, trans=trans, exponent_off=exponent_off, transoffdig=transoffdig)
                        temp2(odefunc, y00, text_file, odefunc=odefunc, time_df=time_df, device=device, exponent_f=exponent_f)

                        text_file.close()
        return phase2_model, odefunc
    else: 
        print('Loading ', load_phase2_path)
        saved_temp = torch.load(load_phase2_path)['state_dict']
        phase2_model.load_state_dict(saved_temp)
        print('Sanity check phase2: ')
        tr_res = test_ce(-1, SingleOutputWrapper(phase2_model), trainloader, device, nn.CrossEntropyLoss(), 110, '')
        te_res = test_ce(-1, SingleOutputWrapper(phase2_model), testloader, device, nn.CrossEntropyLoss(), 110, '')
        print('itr = ', itr, 'Train Acc, Loss', tr_res['acc'], tr_res['loss'])
        print('itr = ', itr, 'Test Acc, Loss', te_res['acc'], te_res['loss'])
        return phase2_model, phase2_model.ode_block.odefunc

train_feature_loader = DataLoader(sst2_train_feature_set,
    batch_size=PHASE2_BS,
    shuffle=True, num_workers=2
)
test_feature_loader = DataLoader(sst2_test_feature_set,
    batch_size=PHASE2_BS,
    shuffle=False, num_workers=2
)

phase2_model, odefunc = phase2(
    bridge_768_64=list(phase1_model)[0],
    trainloader=train_feature_loader,
    testloader=test_feature_loader, 
    ODE_FC_save_folder=f'{LOG_PATH}/{EXP_NAME}',
    fc_layer=list(phase1_model)[1],
    load_phase2_path = f'{LOG_PATH}/{EXP_NAME}/phase2model_19.pth',
)
