import torch
import torchvision.models as models

from ImageReconstruction.model import ContentVGG19

class StyleVGG19(ContentVGG19):
    def __init__(self, device , reconstruction_layers, is_local=False):
        super().__init__(device=device, is_local=is_local)
        self.reconstruction_layers = [self.conv_layers[layer] for layer in reconstruction_layers]

    def forward(self, x):
        model = self.model # for readability
        feature_maps = []
        for i, layer in enumerate(model.children()):
            x = layer(x)
            if i in self.reconstruction_layers:
                feature_maps.append(x)
        return feature_maps