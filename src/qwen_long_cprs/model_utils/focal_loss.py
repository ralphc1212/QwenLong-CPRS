import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='none'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        loss_fct = CrossEntropyLoss(reduction = 'none', weight=self.weight)
        ce_loss = loss_fct(input, target) # (B, 1) loglikelihood
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss