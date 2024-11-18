import numpy as np
import torch
from data.data_utils import divide_3d_image_into_patches, combine_patches_into_3d_image
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis, binarize_image_pp

class CCTAPipeline:
    def __init__(self, model, block_size):
        self.model = model
        self.block_shape = (block_size,) * 3
    
    def __call__(self, ccta_scan):
        prediction = self.process_ccta_scan(ccta_scan)
        return prediction

    def process_ccta_scan(self, ccta_scan: np.array):
        ccta_scan = prepare_image_for_network_input(ccta_scan)
        patches = divide_3d_image_into_patches(ccta_scan, block_shape=self.block_shape)
        patch_shape = patches.shape

        prediction_patches = np.zeros((patch_shape))
        for i in range(patch_shape[0]):
            for j in range(patch_shape[1]):
                for k in range(patch_shape[2]):
                    patch_counter += 1
                    current_image_patch = patches[i, j, k]

                    current_image_patch = prepare_image_for_network_input(current_image_patch)

                    with torch.no_grad():
                        current_prediction_patch = self.model(current_image_patch)
                    
                    prediction_patches[i,j,k] = current_prediction_patch
        
        combined_prediction = combine_patches_into_3d_image(prediction_patches)
        combined_prediction = binarize_image_pp(combined_prediction)
        combined_prediction = prepare_image_for_analysis(combined_prediction)

        assert combined_prediction.shape == [275, 512, 512]