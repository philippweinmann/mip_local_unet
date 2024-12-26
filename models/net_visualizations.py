from matplotlib import pyplot as plt
from data.visualizations import display3DImageMaskTuple
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