from data.visualizations import visualize_3d_matrices
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis

def three_d_visualize_model_progress(model, get_image_fct):
    original_img, original_mask = get_image_fct()

    pred = model(prepare_image_for_network_input(original_img))

    visualize_3d_matrices(original_img, original_mask, prepare_image_for_analysis(pred))

    return original_mask, prepare_image_for_analysis(pred)