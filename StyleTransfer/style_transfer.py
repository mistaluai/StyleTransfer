import torch

from StyleTransfer.style_transferer import StyleTransferer
from utils.image_plotter import ImagePlotter
from utils.image_processor import ImageProcessor

device = 'cuda'
processor = ImageProcessor(device)

target_content = processor.load_image('./How-to-find-beautiful-landscapes-4.jpg')
target_style = processor.load_image('./starry_night.jpg')

# result = torch.rand(target_content.size(), requires_grad=True, device=device)
result = target_content.clone().detach().contiguous().requires_grad_(True)

transferer = StyleTransferer(device=device, is_local=False)

result, outputs = transferer.transfer(target_content, target_style, result, epochs=100, alpha=1, beta=1000000)

images = {
'Generated Image': processor.tensor_to_image(result),
'Ground Truth Style': processor.tensor_to_image(target_style),
'Ground Truth Content': processor.tensor_to_image(target_content),
}


for i, image in enumerate(outputs):
    if (i+1) % 100 == 0:
        s = f'Generation step {i+1}'
        images[s] = processor.tensor_to_image(image)
name = 'StyleTransfer_StarryNightFordCar'
plotter = ImagePlotter(data=images, gif_data=outputs, output_path='./st_outputs', name=name)