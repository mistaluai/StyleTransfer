from StyleReconstruction.model import StyleVGG19
from tqdm import tqdm
import torch.optim as optim

from StyleReconstruction.style_loss import StyleLoss


class StyleReconstructor:
    def __init__(self, device, is_local, reconstruction_layers=None, layers_weights=None):
        if layers_weights is None:
            layers_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        if reconstruction_layers is None:
            reconstruction_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.loss = StyleLoss(layers_weights)
        self.device = device
        self.reconstruction_layers = reconstruction_layers
        self.layers_weights = layers_weights
        self.model = StyleVGG19(device=device, reconstruction_layers=reconstruction_layers, is_local=is_local)


    def reconstruct(self, target_image, result_image, epochs=100):
        optimizer = optim.LBFGS([result_image])
        model, device = self.model, self.device
        target_image = target_image.detach()
        outputs = []
        pbar = tqdm(range(epochs), desc='Reconstruction', unit='epoch')
        for epoch in pbar:
            def closure():
                optimizer.zero_grad()
                target_feature_maps = model(target_image)
                result_feature_maps = model(result_image)
                loss = self.loss(target_feature_maps, result_feature_maps)
                loss.backward()
                return loss

            optimizer.step(closure)
            outputs.append(result_image.clone().detach())
            pbar.set_description(
                desc=f'Reconstruction [result mean:{result_image.mean().item():0.4f}|target mean:{target_image.mean().item():0.4f}]')
