import torch
from utils.image_plotter import ImagePlotter
from utils.image_processor import ImageProcessor
from ImageReconstruction.image_reconstructor import ImageReconstructor

device = 'mps'
processor = ImageProcessor(device)

target_image = processor.load_image('./images/ford.jpg')
result_image = torch.rand(target_image.size(), requires_grad=True, device=device)

reconstructor = ImageReconstructor(device, is_local=True, reconstruction_layer='conv3_3')

result_image, outputs = reconstructor.reconstruct(target_image, result_image, epochs=10)
images = {
'Generated Image': processor.tensor_to_image(result_image),
'Ground Truth Image': processor.tensor_to_image(target_image)
}


for i, image in enumerate(outputs):
    if (i+1) % 10 == 0:
        s = f'Generation step {i+1}'
        images[s] = processor.tensor_to_image(image)

plotter = ImagePlotter(images)