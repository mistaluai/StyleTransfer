import torch
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self, mean=None, std=None, device='cuda'):
        super(Normalization, self).__init__()
        if mean is None:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        if std is None:
            std = torch.tensor([0.229, 0.224, 0.225], device=device)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std