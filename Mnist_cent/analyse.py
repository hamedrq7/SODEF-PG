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

do_wandb = True
if do_wandb:
    wandb.init(project="SODEF-MNIST", name=f'anls-MNIST-64D-CenterLoss-FCinit_cent_weight')

table = wandb.Table(columns=["experiment", "raw_test_loss", "denoised_test_loss", "raw_test_acc", "denoised_test_acc", "no_ode_raw_test_loss", "no_ode_denoised_test_loss", "no_ode_raw_test_acc", "no_ode_denoised_test_acc", "w_ode_raw_test_loss", "w_ode_denoised_test_loss", "w_ode_raw_test_acc", "w_ode_denoised_test_acc"])

for exp in [[0.001, 0.001, 1.0], [0.001, 0.0, 20.0], [0.001, 0.0, 1.0]]:
    cent_weight = exp[0]
    cent_lr = exp[1]
    rad = exp[2]
    exp_name = f'cw_{cent_weight}-clr_{cent_lr}-rad{rad}'
    # device = torch.device("cuda:0")
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    best_acc = 0
    start_epoch = 0
    print(exp_name)


    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    base_data_path = f'./data_cent_init_{exp_name}'
    os.makedirs(base_data_path, exist_ok=True)

    fc_dim = 64
    folder_savemodel = f'./EXP/orth_MNIST_resnet_final_{exp_name}'
    os.makedirs(folder_savemodel, exist_ok=True)


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

    act = torch.sin
    act2 = torch.nn.functional.relu
    fc_dim = 64

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



    class Flatten(nn.Module):

        def __init__(self):
            super(Flatten, self).__init__()

        def forward(self, x):
            shape = torch.prod(torch.tensor(x.shape[1:])).item()
            return x.view(-1, shape)


    class MLP_OUT(nn.Module):

        def __init__(self):
            super(MLP_OUT, self).__init__()
            self.fc0 = nn.Linear(fc_dim, 10)

        def forward(self, input_):
            h1 = self.fc0(input_)
            return h1


    endtime = 5
    layernum = 0

    tol = 1e-5
    savefolder_fc = f'./EXP/orth_resnetfct5_15_cent_{exp_name}/'


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
                break

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
            break 

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


    plot_loss_diff_histogram_trimmed(
        loss_diffs_pgd_w_ode,
        save_path="loss_diff_pgd_w_ode.png",
        title="Adv Data (pgd w/ode): Raw − Denoised Loss"
    )

    # PGD generation w.o./ODE
    w_ode_raw_test_loss, w_ode_raw_test_acc, w_ode_denoised_test_loss, w_ode_denoised_test_acc, loss_diffs_pgd_w_o_ode = evaluate_pgd(testloader, combined_model, no_ode_in_pgd_generation=True)


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

    # feats_before, feats_after, labels = collect_feats(testloader, combined_model)

    @torch.no_grad()
    def integrate_to_long_T(model, z0, T_long):
        """
        z0: (N, d)
        """
        return model.ode_block(z0, t=T_long)
    # T_long = 20.0
    # zT = integrate_to_long_T(combined_model, feats_before, T_long)

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

    # num_classes = 10
    # stable_points = extract_stable_points_per_class(
    #     zT, labels, num_classes
    # )

    # print(estimate_basin_stats_relative(combined_model, stable_points))
    table.add_data(exp_name, raw_test_loss, denoised_test_loss, raw_test_acc, denoised_test_acc, no_ode_raw_test_loss, no_ode_denoised_test_loss, no_ode_raw_test_acc, no_ode_denoised_test_acc, w_ode_raw_test_loss, w_ode_denoised_test_loss, w_ode_raw_test_acc, w_ode_denoised_test_acc)

wandb.log({
    "ABC_by_experiment": table,
    "ABC_barplot": wandb.plot.bar(
        table,
        "experiment",
        ["raw_test_loss", "denoised_test_loss", "raw_test_acc", "denoised_test_acc", "no_ode_raw_test_loss", "no_ode_denoised_test_loss", "no_ode_raw_test_acc", "no_ode_denoised_test_acc", "w_ode_raw_test_loss", "w_ode_denoised_test_loss", "w_ode_raw_test_acc", "w_ode_denoised_test_acc"],
        title="A / B / C per Experiment"
    )
})
