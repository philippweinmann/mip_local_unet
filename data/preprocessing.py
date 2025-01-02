from data_generation.config import mean_voxel_intensity
from data.data_utils import shift_mean

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image
    
def min_max_normalization(image, min_value, max_value):
    # min and max value come from clipping
    
    # normalize the image from 0 to 1
    image = (image - min_value) / (max_value - min_value)
    
    return image

def preprocess_ccta_scan(image):
    min_value = -600
    max_value = 1000
    # image = shift_mean(image, mean_voxel_intensity)
    image = clip_scans(image, min_value, max_value)

    # normalize the image from 0 to 1
    image = min_max_normalization(image, min_value, max_value)
    return image