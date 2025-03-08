import torch
import torch.nn as nn

class Normalization(nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalization, self).__init__()
        if mean is None:
            mean = torch.tensor([0.485, 0.456, 0.406])
        if std is None:
            std = torch.tensor([0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std