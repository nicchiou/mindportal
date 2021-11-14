import torch
from torch import nn


# Focal loss implementation based on the paper: https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        bce_loss = self.loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss
