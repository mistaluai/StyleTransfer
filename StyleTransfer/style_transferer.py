import torch

from ImageReconstruction.content_loss import ContentLoss
from StyleReconstruction.style_loss import StyleLoss
from StyleTransfer.model import StyleTransfer
import torch.optim as optim
from tqdm import tqdm

class StyleTransferer():
    def __init__(self, device ,content_layer='conv4_2', style_weights=None, style_layers=None, is_local=True):
        if style_layers is None:
            style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        if style_weights is None:
            style_weights = [1/len(style_layers)] * len(style_layers)

        self.style_loss = StyleLoss(style_weights)
        self.content_loss = ContentLoss()
        self.device = device
        self.model = StyleTransfer(device=device, content_layer=content_layer, style_layers=style_layers, is_local=is_local)

    def transfer(self, content, style, result_image, epochs=1000, alpha=1, beta=1000):
        model, style_loss, content_loss = self.model, self.style_loss, self.content_loss
        optimizer = optim.LBFGS([result_image])
        content = content.detach()
        style = style.detach()

        outputs = []
        pbar = tqdm(range(epochs), desc='Styling', unit='epoch')
        for epoch in pbar:
            def closure():
                self.clamp(result_image)

                optimizer.zero_grad()
                target_content_feature_map, _ = model(content)
                _, style_feature_maps = model(style)

                result_content_feature_map, result_style_feature_maps = model(result_image)
                loss_c = content_loss(target_content_feature_map, result_content_feature_map)
                loss_s = style_loss(style_feature_maps, result_style_feature_maps)

                loss = alpha * loss_c + beta * loss_s
                loss.backward()
                return loss

            self.clamp(result_image)
            optimizer.step(closure)
            outputs.append(result_image.clone().detach())

        self.clamp(result_image)

        return result_image, outputs


    def clamp(self, image):

        with torch.no_grad():
            image.clamp_(0, 1)

        return image
