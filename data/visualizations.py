# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
def visualize_3d_matrices(matrices, titles, global_title = None, show_axis=True):
    amt_images = len(matrices)
    if amt_images > 9:
        raise ValueError("Can only visualize up to 9 images at once.")
    
    # we want a max of 3 columns.
    amt_cols = min(3, amt_images)
    amt_rows = int(np.ceil(amt_images / amt_cols))
    
    figsize=(amt_cols * 10, amt_rows * 10)

    fig = plt.figure(figsize=figsize)
    if global_title:
        fig.suptitle(global_title, y=0.95, fontsize=20)

    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        # Iterating over the grid returns the Axes.
        x, y, z = np.where(matrix >= 0.5)
        ax = fig.add_subplot(amt_rows, amt_cols, i + 1, projection='3d')
        ax.scatter(x, y, z, c='red', marker='o', s=0.7, alpha=0.2)

        # Set labels even if we don't show them
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([0, matrix.shape[0]])
        ax.set_ylim([0, matrix.shape[1]])
        ax.set_zlim([0, matrix.shape[2]])

        if not show_axis:
            ax.axis('off')

        ax.set_title(title, fontsize=18)

    plt.show()
    
def visualize_model_confidence(prediction: np.array):
    flattended_pred = prediction.flatten()
    count_0s = np.sum(flattended_pred == 0)
    count_1s = np.sum(flattended_pred == 1)
    neither_count = np.sum((flattended_pred != 0) & (flattended_pred != 1))
    
    print(f"{neither_count}: not 0, nor 1\n{count_0s}: 0s\n{count_1s}: 1s")
    
    plt.hist(flattended_pred, bins=50)
    
    plt.xlabel("bin means")
    plt.ylabel("amount of elements in bin")
    plt.title("confidence visualization of the model")
    
    plt.show()