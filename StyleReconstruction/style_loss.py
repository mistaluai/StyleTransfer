import torch
from torch import nn
import torch.nn.functional as F
from utils.gram_matrix import gram_matrix


class StyleLoss(nn.Module):
    def __init__(self, style_weights):
        super().__init__()
        self.style_weights = style_weights

    def forward(self, target_feature_maps, result_feature_maps):
        total_loss = 0

        for idx, _ in enumerate(target_feature_maps):
            G = gram_matrix(result_feature_maps[idx])
            A = gram_matrix(target_feature_maps[idx])
            loss = F.mse_loss(G, A)
            loss *= self.style_weights[idx]
            total_loss += loss
        return total_loss