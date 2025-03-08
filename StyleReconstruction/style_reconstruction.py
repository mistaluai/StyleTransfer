import torch

from StyleReconstruction.style_reconstructor import StyleReconstructor
from utils.image_plotter import ImagePlotter
from utils.image_processor import ImageProcessor
from ImageReconstruction.image_reconstructor import ImageReconstructor

device = 'mps'
processor = ImageProcessor(device)

target_style = processor.load_image('./images/starry_night.jpg')
result_style = torch.rand(target_style.size(), requires_grad=True, device=device)

reconstructor = StyleReconstructor(device=device, is_local=True)

result_style, outputs = reconstructor.reconstruct(target_style, result_style, epochs=20)

images = {
'Generated Style': processor.tensor_to_image(result_style),
'Ground Truth Style': processor.tensor_to_image(target_style)
}


for i, image in enumerate(outputs):
    if (i+1) % 2 == 0:
        s = f'Generation step {i+1}'
        images[s] = processor.tensor_to_image(image)
name = 'StyleReconstruction_StarryNight'
plotter = ImagePlotter(data=images, gif_data=outputs, output_path='./reconstruction_outputs', name=name)