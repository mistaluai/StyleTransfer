import torch
import torch.nn as nn
import torchvision.models as models

class CustomVGG19(nn.Module):

    def __init__(self, reconstruction_layers=None):
        super().__init__()
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

        self.reconstruction_layers = [conv_layers[layer] for layer in reconstruction_layers]
        print(self.model)
        print(reconstruction_layers)
        print(self.reconstruction_layers)


    def __prepare_model(self, model_path=None):
        model = models.vgg19()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        # model = nn.Sequential(*list(model.children())[:-2])
        model = model.features
        return model

    def forward(self, x):
        pass


CustomVGG19 = CustomVGG19()