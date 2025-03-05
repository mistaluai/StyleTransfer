import torch
import torch.nn as nn
import torchvision.models as models

class CustomVGG19(nn.Module):

    def __init__(self,device , reconstruction_layer='conv4_2'):
        super().__init__()
        self.device = device
        self.model = self.__prepare_model('../PretrainedModels/vgg19-dcbb9e9d.pth')

        conv_layers = {
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

        self.reconstruction_layer = conv_layers[reconstruction_layer]

    def __prepare_model(self, model_path=None):
        model = models.vgg19()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        # model = nn.Sequential(*list(model.children())[:-2])
        model = model.features
        return model.to(self.device)

    def forward(self, x):
        model = self.model #for code readability

        for i, layer in enumerate(model.children()):
            x = layer(x)
            if i == self.reconstruction_layer:
                return x