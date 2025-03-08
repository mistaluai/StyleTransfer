import torch
import torch.nn as nn
import torchvision.models as models
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
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv3_4': 16,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv4_3': 23,
            'conv4_4': 25,
            'conv5_1': 28,
            'conv5_2': 30,
            'conv5_3': 32,
            'conv5_4': 34
        }
        self.content_layer = self.conv_layers[content_layer]
        self.style_layers = [self.conv_layers[layer] for layer in style_layers]



    def __prepare_model(self, model_path=None):
        if model_path is not None:
            model = models.vgg19()
            model.load_state_dict(torch.load(model_path))
        else:
            model = models.vgg19(weights='IMAGENET1K_V1')
        model = model.features
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
