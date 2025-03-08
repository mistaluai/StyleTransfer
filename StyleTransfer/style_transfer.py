import torch

from StyleTransfer.style_transferer import StyleTransferer
from utils.image_processor import ImageProcessor

device = 'cuda'
processor = ImageProcessor(device)

target_style = processor.load_image('Villem_Ormisson,_Tartu_vaade_(1937).jpg')
target_content = processor.load_image('Pyramid-of-Khafre-Giza-Egypt.jpg')

result = target_content.clone().detach().contiguous().requires_grad_(True)
result.add(torch.rand(result.size(), requires_grad=True, device=device))

transferer = StyleTransferer(device=device, is_local=False)

result, outputs = transferer.transfer(target_content, target_style, result, epochs=100, alpha=10, beta=50)

images = {
'Generated Image': processor.tensor_to_image(result),
'Ground Truth Style': processor.tensor_to_image(target_style),
'Ground Truth Content': processor.tensor_to_image(target_content),
}


for i, image in enumerate(outputs):
    if (i+1) % 50 == 0:
        s = f'Generation step {i+1}'
        images[s] = processor.tensor_to_image(image)