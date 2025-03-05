import torch
import torch.nn as nn
import torch.optim as optim

from ImageReconstruction.image_plotter import ImagePlotter
from ImageReconstruction.image_processor import ImageProcessor
from ImageReconstruction.image_reconstructor import ImageReconstructor

device = 'mps'
processor = ImageProcessor(device)

target_image = processor.load_image('./images/ford.jpg')
result_image = torch.rand(target_image.size()).to(device)

reconstructor = ImageReconstructor(device)

result_image, outputs = reconstructor.reconstruct(target_image, result_image)

plotter = ImagePlotter({'noise': processor.tensor_to_image(result_image), 'target_image': processor.tensor_to_image(target_image)})