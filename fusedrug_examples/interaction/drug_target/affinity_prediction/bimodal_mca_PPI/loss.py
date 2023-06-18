import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor
from typing import Dict


# TODO: consider also adding "MyGammaLoss" which doesn't involve crossentropy term at all
class FocalLoss(nn.Module):
    def __init__(self, weight: float = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 - prob) ** self.gamma) * log_prob, gt, weight=self.weight, reduction=self.reduction,)


loss_funcs = {
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
}


def get_loss_func(name: str, **kwargs: Dict) -> Tensor:
    if name not in loss_funcs:
        raise Exception(f'loss function "{name}" not recognized. Supported options are: {loss_funcs.keys()}')
    func = loss_funcs[name]
    return func(**kwargs)
