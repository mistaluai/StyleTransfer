import matplotlib.pyplot as plt
from PIL import Image

from utils.image_processor import ImageProcessor


class ImagePlotter:
    def __init__(self, data, gif_data,output_path, name, device='mps'):
        self.data = data
        self.gif_data = gif_data
        self.output_path = output_path
        self.name = name
        self.device = device
        self.plot_images()
        self.create_gif()


    def create_gif(self, duration=400, loop=0):
        processor = ImageProcessor(self.device)
        images = [processor.tensor_to_image(tensor=image) for image in self.gif_data]
        output_path = self.output_path + f'/{self.name}.gif'
        images[0].save(
            output_path, save_all=True, append_images=images[1:], duration=duration, loop=loop
        )
        print(f"GIF saved at {output_path}")

    def plot_images(self):
        output_path = self.output_path + f'/{self.name}.png'
        fig, axes = plt.subplots(1, len(self.data), figsize=(len(self.data) * 3, 3))

        if len(self.data) == 1:
            axes = [axes]  # Ensure axes is iterable for a single image

        for ax, (name, img) in zip(axes, self.data.items()):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Save as PNG
        print(f"Plot saved as {output_path}")
        plt.show()
        plt.close(fig)  # Close the figure to free memory