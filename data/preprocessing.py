from data_generation.config import mean_voxel_intensity
from data.data_utils import shift_mean

def preprocess_ccta_scan(image):
    image = shift_mean(image, mean_voxel_intensity)

    return image