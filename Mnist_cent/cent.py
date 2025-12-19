# https://pytorch-ood.readthedocs.io/en/v0.1.9/index.html
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, KeysView, Optional, Tuple, TypeVar, Union
from torch import Tensor
import numpy as np 

def is_known(labels) -> Union[bool, Tensor]:
    """
    :returns: True, if label :math:`>= 0`
    """
    return labels >= 0

def pairwise_distances(x: Tensor, y: Tensor = None) -> Tensor:
    """
    Calculate pairwise squared Euclidean distance by quadratic expansion.

    :param x: is a :math:`N \\times D` matrix
    :param y:  :math:`M \\times D` matrix
    :returns: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

    :see Implementation: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = y.pow(2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

class ClassCenters(nn.Module):
    """
    Several methods for OOD Detection propose to model a center :math:`\\mu_y` for each class.
    These centers are either static, or learned via gradient descent.

    The centers are also known as class proxy, class prototype or class anchor.
    """

    def __init__(self, n_classes: int, n_features: int, fixed: bool = False):
        """

        :param n_classes: number of classes vectors
        :param n_features: dimensionality of the space in which the centers live
        :param fixed: False if the centers should be learnable parameters, True if they should be fixed at their
            initial position
        """
        super(ClassCenters, self).__init__()
        # anchor points are fixed, so they do not require gradients
        self._params = nn.Parameter(torch.randn(size=(n_classes, n_features)))

        if fixed:
            self._params.requires_grad = False

    @property
    def num_classes(self) -> int:
        return self.params.shape[0]

    @property
    def n_features(self) -> int:
        return self.params.shape[1]

    @property
    def params(self) -> nn.Parameter:
        """
        Class centers :math:`\\mu`
        """
        return self._params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: samples
        :returns: pairwise squared distance of samples to each center
        """
        assert x.shape[1] == self.n_features
        return pairwise_distances(x, self.params)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make class membership predictions based on the softmin of the distances to each center.

        :param x: embeddings of samples
        :returns: normalized pairwise distance of samples to each center
        """
        distances = pairwise_distances(x, self.params)
        return nn.functional.softmin(distances, dim=1)

class CenterLoss(nn.Module):
    """
    Generalized version of the Center Loss from the Paper
    *A Discriminative Feature Learning Approach for Deep Face Recognition*.
    For each class, this loss places a center :math:`\\mu_y` in the output space and draws representations of samples
    to their corresponding class centers, up to a radius :math:`r`.

    Calculates

    .. math::
        \\mathcal{L}(x,y) = \\max \\lbrace  d(f(x),\\mu_y) - r , 0 \\rbrace

    where :math:`d` is some measure of dissimilarity, like the squared distance.

    With radius :math:`r=0` and the squared euclidean distance as :math:`d(\\cdot,\\cdot)`, this is equivalent to
    the original center loss, which is also referred to as the *soft-margin loss* in some publications.

    :see Implementation: `GitHub <https://github.com/KaiyangZhou/pytorch-center-loss>`__
    :see Paper: `ECCV 2016 <https://ydwen.github.io/papers/WenECCV16.pdf>`__
    """

    def __init__(
        self,
        n_classes: int,
        n_dim: int,
        magnitude: float = 1.0,
        radius: float = 0.0,
        fixed: bool = False,
    ):
        """
        :param n_classes: number of classes :math:`C`
        :param n_dim: dimensionality of center space :math:`D`
        :param magnitude:  scale :math:`\\lambda` used for center initialization
        :param radius: radius :math:`r` of spheres, lower bound for distance from center that is penalized
        :param fixed: false if centers should be learnable
        """
        super(CenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_dim
        self.magnitude = magnitude
        self.radius = radius
        self._centers = ClassCenters(n_classes=n_classes, n_features=n_dim, fixed=fixed)
        self._init_centers()

    @property
    def centers(self) -> ClassCenters:
        """
        :return: the :math:`\\mu` for all classes
        """
        return self._centers

    def _init_centers(self):
        # In the published code, Wen et al. initialize centers randomly.
        # However, this might bot be optimal if the loss is used without an additional
        # inter-class-discriminability term.
        # The Class Anchor Clustering initializes the centers as scaled unit vectors.
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self._centers._params)
            if not self._centers._params.requires_grad:
                self._centers._params.mul_(self.magnitude)
        # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
        # different then the number of classes
        # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers.params)

    def forward(self, distmat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss. Ignores OOD inputs.

        :param distmat: matrix of distances of each point to each center with shape :math:`B \\times C`.
        :param target: ground truth labels with shape (batch_size).
        :returns: the loss values
        """
        known = is_known(target)

        if known.any():
            distmat = distmat[known]
            target = target[known]
            batch_size = distmat.size(0)

            classes = torch.arange(self.num_classes).long().to(distmat.device)
            target = target.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = target.eq(classes.expand(batch_size, self.num_classes))
            dist = (distmat - self.radius).relu() * mask.float()
            loss = dist.clamp(min=1e-12, max=1e12).mean()
        else:
            loss = torch.tensor(0.0, device=distmat.device)

        return loss


# class CenterLossNormal(nn.Module):
#     """Center loss.
    
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=10, feat_dim=64, use_gpu=True, init_value = None, rad = 1.0):
#         super(CenterLossNormal, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu
#         self.init_value = init_value

#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda()) * rad
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))* rad

#         if self.init_value is not None:

#             with torch.no_grad():
#                 self.centers.copy_(self.init_value)

            
#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

#         return loss

from torch.autograd.function import Function

class CenterLossNormal(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True, init_value = None):
        super(CenterLossNormal, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        if init_value is not None: 
            with torch.no_grad():
                self.centers.copy_(init_value)
        
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None
