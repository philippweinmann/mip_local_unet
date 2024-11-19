from data_generation.config import mean_voxel_intensity
from data.data_utils import shift_mean, clip_scans

def preprocess_ccta_scan(image):
    # image = shift_mean(image, mean_voxel_intensity)
    image = clip_scans(image, -500, 1000)
    return image