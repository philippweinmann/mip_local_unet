import numpy as np
from matplotlib import pyplot as plt

def display3DImageMaskTuple(image, mask, predicted_mask = None):
    images_to_plot = [image, mask]
    titles = ["image", "mask"]

    if predicted_mask is not None:
        images_to_plot.append(predicted_mask)
        titles.append("predicted mask")

    fig = plt.figure(figsize=(10, 10))

    for plt_idx, (image, title) in enumerate(zip(images_to_plot, titles)):
        x, y, z = np.where(image >= 0.5)

        ax = fig.add_subplot(1, 3, plt_idx + 1, projection='3d')

        # Plot the '1's in the image
        # modify alpha for transparency, s for size
        ax.scatter(x, y, z, c='red', marker='o', s=0.5, alpha=0.2)

        # Set labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([0, image.shape[0]])
        ax.set_ylim([0, image.shape[1]])
        ax.set_zlim([0, image.shape[2]])

        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.pause(0.001)