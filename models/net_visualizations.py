from matplotlib import pyplot as plt
import numpy as np
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis


def visualize_model_parameters(model, batch_number):
    for tag, parm in model.named_parameters():
        if parm.grad is not None:
            parm_grad = parm.grad.cpu().numpy().flatten()
            # print(tag)
            # print(parm_grad)
            plt.hist(parm_grad, bins=10)
            plt.title(tag + " gradient, batch number = " + str(batch_number))
            plt.xlabel("bin means")
            plt.ylabel("amount of elements in bin")
            plt.show()
            plt.pause(0.001)


# %%
def display2DImageMaskTuple(image, mask, predicted_mask = None):
    if predicted_mask is not None:
        amt_subplots = 3
    else:
        amt_subplots = 2

    fig, ax = plt.subplots(1, amt_subplots, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')

    if predicted_mask is not None:
        ax[2].imshow(predicted_mask, cmap='gray')
        ax[2].set_title('Predicted Mask')
    
    plt.pause(0.001)

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
        ax.scatter(x, y, z, c='red', marker='o')

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


def two_d_visualize_model_progress(model, get_image_fct):
    original_img, original_mask = get_image_fct()

    pred = model(prepare_image_for_network_input(original_img))

    display2DImageMaskTuple(original_img, original_mask, prepare_image_for_analysis(pred))

    return original_mask, prepare_image_for_analysis(pred)

# %%
def three_d_visualize_model_progress(model, get_image_fct):
    original_img, original_mask = get_image_fct()

    pred = model(prepare_image_for_network_input(original_img))

    display3DImageMaskTuple(original_img, original_mask, prepare_image_for_analysis(pred))

    return original_mask, prepare_image_for_analysis(pred)