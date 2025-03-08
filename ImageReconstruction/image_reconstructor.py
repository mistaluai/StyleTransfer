import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ImageReconstruction.content_loss import ContentLoss
from ImageReconstruction.model import ContentVGG19


class ImageReconstructor:
    def __init__(self, device, is_local, reconstruction_layer='conv4_2'):
        self.device = device
        print(f'Initialized ImageReconstructor with device {self.device}')
        self.model = ContentVGG19(device, is_local=is_local, reconstruction_layer=reconstruction_layer)
        self.loss = ContentLoss()


    def reconstruct(self, target_image, result_image, epochs=100):
        target_image = target_image.detach()
        optimizer = optim.LBFGS([result_image])
        model, device = self.model, self.device
        outputs = []
        pbar = tqdm(range(epochs), desc='Reconstruction', unit='epoch')
        for epoch in pbar:

            def closure():
                optimizer.zero_grad()
                target_features = model(target_image)
                result_features = model(result_image)
                loss = self.loss(target_features, result_features)
                loss.backward()
                return loss

            optimizer.step(closure)
            outputs.append(result_image.clone().detach())
            pbar.set_description(desc=f'Reconstruction [result mean:{result_image.mean().item():0.4f}|target mean:{target_image.mean().item():0.4f}]')

        return result_image, outputs




