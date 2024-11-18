import numpy as np
import torch
from data.data_utils import divide_3d_image_into_patches, combine_patches_into_3d_image, pad_image
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis, binarize_image_pp

class CCTA_scan:
    def __init__(self, ccta_scan: np.array, block_size):
        self.ccta_scan = ccta_scan
        self.original_shape = ccta_scan.shape

        self.prediction = None

        # adds padding at the end
        self.padded_image, self.z_pad_length = pad_image(self.ccta_scan, patch_size=block_size)
        assert np.all(self.padded_image[self.padded_image.shape[0] - self.z_pad_length:] == 0)

    def get_unpadded_prediction(self, prediction):
        assert prediction.shape == self.padded_image.shape
        self.prediction = prediction

        # assuming the padding was added at the end
        self.unpadded_prediction = prediction[0:prediction.shape[0]-self.z_pad_length]

        return self.unpadded_prediction



class CCTAPipeline:
    def __init__(self, model, block_size):
        self.model = model
        self.block_shape = (block_size,) * 3
    
    def __call__(self, ccta_scan):
        prediction = self.process_ccta_scan(ccta_scan)
        return prediction

    def process_ccta_scan(self, ccta_scan: np.array):
        ccta_scan_obj = CCTA_scan(ccta_scan, block_size=self.block_shape[0])
        padded_scan = ccta_scan_obj.padded_image

        patches = divide_3d_image_into_patches(padded_scan, block_shape=self.block_shape)
        patch_shape = patches.shape

        prediction_patches = np.zeros((patch_shape))
        patch_counter = 0
        for i in range(patch_shape[0]):
            for j in range(patch_shape[1]):
                for k in range(patch_shape[2]):
                    patch_counter += 1
                    print(f"processing patch {patch_counter} / {patch_shape[0] * patch_shape[1] * patch_shape[2]}")
                    current_image_patch = patches[i, j, k]

                    current_image_patch = prepare_image_for_network_input(current_image_patch)

                    with torch.no_grad():
                        current_prediction_patch = self.model(current_image_patch)
                    
                    prediction_patches[i,j,k] = current_prediction_patch
        
        combined_prediction = combine_patches_into_3d_image(prediction_patches)
        combined_prediction = binarize_image_pp(combined_prediction)
        combined_prediction = prepare_image_for_analysis(combined_prediction)

        combined_prediction = ccta_scan_obj.get_unpadded_prediction(combined_prediction)

        assert combined_prediction.shape == ccta_scan_obj.original_shape, f"combined_prediction.shape: {combined_prediction.shape}"

        return combined_prediction