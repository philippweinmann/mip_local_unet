from data_generation.config import mean_voxel_intensity
from data.data_utils import shift_mean, clip_scans

def preprocess_ccta_scan(image):
    min_value = -500
    max_value = 1000
    # image = shift_mean(image, mean_voxel_intensity)
    image = clip_scans(image, min_value, max_value)

    # normalize the image from 0 to 1
    image = (image - min_value) / (max_value - min_value)
    return image