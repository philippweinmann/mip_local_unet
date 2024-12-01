# %%
import numpy as np
from skimage.util import view_as_blocks
from scipy.ndimage import zoom
import nibabel as nib
import random

target_voxel_spacing = [0.5, 0.5, 0.5]

def get_voxel_spacing(image_fp):
    image = nib.load(image_fp)
    (img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing) = image.header.get_zooms()[:3]
    
    return [img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing]

def resample_image(image, original_spacing, target_spacing = target_voxel_spacing):
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    return resampled_image

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image

def min_max_normalize(image, min_value, max_value):
    image = (image - min_value) / (max_value - min_value)

    return image

def pad_image(image, patch_size):
    shape = image.shape
    print(shape)

    padded_image = image
    pad_lengths = []
    for idx, shape_dim in enumerate(shape):
        rest = shape_dim % patch_size

        if rest == 0:
            print(f"no padding required for dim {idx}. Original image shape: {image.shape}")
            continue

        # we need to add that many slices
        pad_length = patch_size - rest
        pad_lengths.append(pad_length)

        # Pad the array with zeros along the first dimension, the zeroes are added at the end
        pad_width = [(0, pad_length) if i == idx else (0, 0) for i in range(len(shape))]
        padded_image = np.pad(padded_image, pad_width, mode='constant')

        assert np.all(padded_image[-pad_length:0] == 0)

    return padded_image, pad_lengths
# %%

def divide_3d_image_into_patches(image_3d, block_shape):
    # print("dividing image into patches")
    # print(f"image shape: {image_3d.shape}, block shape: {block_shape}")
    patches = view_as_blocks(image_3d, block_shape).squeeze()

    return patches

# patches = divide_3d_image_into_patches(image_3d=np.zeros((320, 512, 512)), block_shape = (64, 64, 64))
# %%
# print(patches.shape)
# %%
def combine_patches_into_3d_image(patches):
    patch_shape = patches.shape
    block_shape = (patch_shape[3], patch_shape[4], patch_shape[5])

    reconstructed_image = np.zeros((patch_shape[0] * block_shape[0],
                                patch_shape[1] * block_shape[1],
                                patch_shape[2] * block_shape[2]))

    # Fill in the reconstructed image with the patches
    # just do these 3 for loops during training as well
    for i in range(patch_shape[0]):
        for j in range(patch_shape[1]):
            for k in range(patch_shape[2]):
                current_patch = patches[i, j, k]
                reconstructed_image[
                    i * block_shape[0]:(i + 1) * block_shape[0],
                    j * block_shape[1]:(j + 1) * block_shape[1],
                    k * block_shape[2]:(k + 1) * block_shape[2]
                ] = current_patch

    # Verify if the reconstructed image matches the original dimensions
    print(reconstructed_image.shape)

    return reconstructed_image
# %%
def get_padded_patches(image, mask, patch_size):
    block_shape = (patch_size, patch_size, patch_size)

    image, _ = pad_image(image, patch_size=patch_size)
    mask, _ = pad_image(mask, patch_size=patch_size)

    image_patches = divide_3d_image_into_patches(np.squeeze(image), block_shape=block_shape)
    mask_patches = divide_3d_image_into_patches(np.squeeze(mask), block_shape=block_shape)

    return image_patches, mask_patches

def clip_scans(image, min_value, max_value):
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image

def shift_mean(image, average_mean):
    current_mean = np.mean(image)
    difference = average_mean - current_mean
    adjusted_image = image + difference

    return adjusted_image

def calculate_voxel_intensities_of_the_masked_area(patients):
    bin_edges = np.arange(-2000, 2000, 100)
    total_counts = np.zeros(len(bin_edges) - 1)

    amt_images = len(patients)
    for image_idx, patient in enumerate(patients):
        image, mask = patient.get_image_mask_tuple()

        print(f"processing image {image_idx + 1} / {amt_images}")
        masked_image = image[mask > 0]
        counts, _ = np.histogram(masked_image, bin_edges)
        total_counts += counts
        
    return total_counts, bin_edges

    

def calculate_voxel_intensities_of_patches(patients, block_shape = (64, 64, 64)):
    bin_edges = np.arange(-2000, 2000, 100)
    total_counts_patch_with_coronary_arteries = np.zeros(len(bin_edges) - 1)
    total_counts_patch_without_coronary_arteries = np.zeros(len(bin_edges) - 1)

    amt_images = len(patients)
    for image_idx, patient in enumerate(patients):
        print(f"processing image {image_idx + 1} / {amt_images}")
        image, mask = patient.get_image_mask_tuple()
        image, _ = pad_image(image, patch_size=block_shape[0])
        mask, _ = pad_image(mask, patch_size=block_shape[0])

        image_patches = divide_3d_image_into_patches(image, block_shape)
        mask_patches = divide_3d_image_into_patches(mask, block_shape)

        patch_shapes = image_patches.shape
        for i in range(patch_shapes[0]):
            for j in range(patch_shapes[1]):
                for k in range(patch_shapes[2]):
                    current_image_patch = image_patches[i, j, k]
                    current_mask_patch = mask_patches[i, j, k]

                    counts, _ = np.histogram(current_image_patch, bin_edges)

                    if np.any(current_mask_patch):
                        total_counts_patch_with_coronary_arteries += counts
                    else:
                        total_counts_patch_without_coronary_arteries += counts
        
    return total_counts_patch_with_coronary_arteries, total_counts_patch_without_coronary_arteries, bin_edges


def get_preprocessed_patches(patches_folder):
    patch_fps = list(patches_folder.iterdir())

    patch_fps = [file for file in patch_fps if "ipynb_checkpoints" not in str(file)]

    print("amt of detected patch files: ", len(patch_fps))
    # setting random seed for reproducibility
    random.Random(42).shuffle(patch_fps)

    return patch_fps


def get_image_mask_from_patch_fp(patch_fp):
    patch = np.load(patch_fp)
    image = patch["image"]
    mask = patch["mask"].astype(np.bool_)

    return image, mask
