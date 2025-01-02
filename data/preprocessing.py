from data_generation.config import mean_voxel_intensity
from data.data_utils import shift_mean
from scipy.ndimage import zoom

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image
    
def min_max_normalization(image, min_value, max_value):
    # min and max value come from clipping
    
    # normalize the image from 0 to 1
    image = (image - min_value) / (max_value - min_value)
    
    return image

def center_image(image):
    # I know this isn't great. But since we're using Instance norms anyway, let's just do it.
    avg = image.mean()
    image -= avg
    
    return image

def resample_image(image, original_spacing, target_spacing):
    
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    
    return resampled_image

def preprocess_ccta_scan(image, mask, original_voxel_spacing):
    min_value = -600
    max_value = 1000
    # image = shift_mean(image, mean_voxel_intensity)
    image = clip_scans(image, min_value, max_value)

    # normalize the image from 0 to 1
    image = min_max_normalization(image, min_value, max_value)
    
    # center the image
    image = center_image(image)
    
    # fix the voxel spacing
    image = resample_image(image, original_voxel_spacing, target_spacing = [0.3, 0.3, 0.3])
    mask = resample_image(mask, original_voxel_spacing, target_spacing = [0.3, 0.3, 0.3])
    
    return image, mask