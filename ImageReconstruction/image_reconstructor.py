import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from ImageReconstruction.model import CustomVGG19


class ImageReconstructor:
    def __init__(self, device, is_local, reconstruction_layer='conv4_2'):
        self.device = device
        print(f'Initialized ImageReconstructor with device {self.device}')
        self.model = CustomVGG19(device, is_local=is_local, reconstruction_layer=reconstruction_layer)


    def reconstruct(self, target_image, result_image, epochs=100):
        result_image.requires_grad = True
        optimizer = optim.LBFGS([result_image])
        model, device = self.model, self.device
        outputs = []
        pbar = tqdm(range(epochs), desc='Reconstruction', unit='epoch')
        for epoch in pbar:

            def closure():
                optimizer.zero_grad()
                target_features = model(target_image)
                result_features = model(result_image)
                loss = 0.5 * F.mse_loss(target_features, result_features)
                loss.backward()
                return loss

            optimizer.step(closure)
            outputs.append(result_image.clone().detach())
            pbar.set_description(desc=f'Reconstruction [result mean:{result_image.mean().item():0.4f}|target mean:{target_image.mean().item():0.4f}]')

        return result_image, outputs




