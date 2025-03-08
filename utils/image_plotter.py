import matplotlib.pyplot as plt


class ImagePlotter:
    def __init__(self, data):
        self.data = data
        self.plot_images()

    def plot_images(self):
        fig, axes = plt.subplots(1, len(self.data), figsize=(len(self.data) * 3, 3))

        if len(self.data) == 1:
            axes = [axes]  # Ensure axes is iterable for a single image

        for ax, (name, img) in zip(axes, self.data.items()):
            ax.imshow(img)
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()
        plt.show()