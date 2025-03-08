from torch import nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_features, result_features):
        return 0.5 * F.mse_loss(target_features, result_features)