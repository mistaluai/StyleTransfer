import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from ImageReconstruction.model import CustomVGG19


class ImageReconstructor:
    def __init__(self, device):
        self.device = device
        print(f'Initialized ImageReconstructor with device {self.device}')
        self.model = CustomVGG19(device)


    def reconstruct(self, target_image, result_image, epochs=100):
        result_image.requires_grad = True
        optimizer = optim.LBFGS([result_image])
        model, device = self.model, self.device
        outputs = []
        for epoch in tqdm(range(epochs), desc='Reconstruction'):



            def closure():
                optimizer.zero_grad()
                target_features = model(target_image)
                result_features = model(result_image)
                loss = 0.5 * F.mse_loss(target_features, result_features)
                loss.backward()
                return loss

            optimizer.step(closure)

        return result_image, outputs




