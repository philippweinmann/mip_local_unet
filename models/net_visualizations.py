from data.visualizations import visualize_3d_matrices
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis

def three_d_visualize_model_progress(model, get_image_fct):
    # not tested
    original_img, original_mask = get_image_fct()

    pred = model(prepare_image_for_network_input(original_img))

    matrices = [original_img, original_mask, prepare_image_for_analysis(pred)]
    titles = ["Original Image", "Original Mask", "Prediction"]
    visualize_3d_matrices(matrices, titles, global_title="Model Progress")

    return original_mask, prepare_image_for_analysis(pred)