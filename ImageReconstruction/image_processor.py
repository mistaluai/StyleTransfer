from PIL import Image
from torchvision.transforms import v2
import torch

class ImageProcessor:
    def __init__(self, device, size=224):
        self.device = device
        print(f'Initialized ImageProcessor with device {self.device}')
        self.size = size

    def __image_transforms(self, image):
        transforms = v2.Compose([
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        return transforms(image)

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = self.__image_transforms(image)
        image = image.unsqueeze(0) #add batch size of 1
        return image.to(self.device, torch.float)

    def tensor_to_image(self, tensor):
        transform = v2.ToPILImage()
        tensor = tensor.squeeze(0)
        return transform(tensor)

