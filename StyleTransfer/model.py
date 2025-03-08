import torch
import torch.nn as nn
import torchvision.models as models

from StyleTransfer.normalizer import Normalization


class StyleTransfer(nn.Module):
    def __init__(self, device , content_layer='conv4_2', style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], is_local=False):
        super().__init__()
        self.device = device
        print(f'Initialized Model with device {self.device}')
        if is_local:
            self.model = self.__prepare_model('../PretrainedModels/vgg19-dcbb9e9d.pth')
        else:
            self.model = self.__prepare_model()
        self.conv_layers = {
            'norm':0,
            'conv1_1': 1,
            'conv1_2': 3,
            'conv2_1': 6,
            'conv2_2': 8,
            'conv3_1': 11,
            'conv3_2': 13,
            'conv3_3': 15,
            'conv3_4': 17,
            'conv4_1': 20,
            'conv4_2': 22,
            'conv4_3': 24,
            'conv4_4': 26,
            'conv5_1': 29,
            'conv5_2': 31,
            'conv5_3': 33,
            'conv5_4': 35
        }
        self.content_layer = self.conv_layers[content_layer]
        self.style_layers = [self.conv_layers[layer] for layer in style_layers]
        print(self.content_layer, self.style_layers)



    def __prepare_model(self, model_path=None):
        if model_path is not None:
            model = models.vgg19()
            model.load_state_dict(torch.load(model_path))
        else:
            model = models.vgg19(weights='IMAGENET1K_V1')
        model = model.features
        model = nn.Sequential(Normalization(), *list(model.children())[:])
        return model.to(self.device).eval()

    def forward(self, x):
        model = self.model # for readability
        style_feature_maps = []
        content_feature_map = None
        for i, layer in enumerate(model.children()):
            x = layer(x)
            if i in self.style_layers:
                style_feature_maps.append(x)
            if i == self.content_layer:
                content_feature_map = x

        return content_feature_map, style_feature_maps