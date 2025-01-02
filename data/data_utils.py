# %%
import numpy as np
from skimage.util import view_as_blocks
from scipy.ndimage import zoom
import nibabel as nib
import random
import os
import torch
from models.net_utils import prepare_image_for_network_input, prepare_image_for_analysis

target_voxel_spacing = [0.5, 0.5, 0.5]

def get_voxel_spacing(image_fp):
    image = nib.load(image_fp)
    (img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing) = image.header.get_zooms()[:3]
    
    return [img_x_dim_spacing, img_y_dim_spacing, img_z_dim_spacing]

def resample_image(image, original_spacing, target_spacing = target_voxel_spacing):
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    resampled_image = zoom(image, zoom_factors, order=1)  # Linear interpolation
    return resampled_image

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
    # print(reconstructed_image.shape)

    return reconstructed_image

def get_all_patches_with_certain_idx(ids, preprocessed_patches):
    '''
    returns a list of lists, each list has all the patches that have the same id as an item in ids.
    meant to be used to get validation or test sets.
    '''
    idx_patches = [[] for _ in range(len(ids))]
    
    for preprocessed_patch in preprocessed_patches:
        current_patch_index = get_idx_from_patch_fp(preprocessed_patch)
        
        for counter, id in enumerate(ids):
            if current_patch_index == id:
                idx_patches[counter].append(preprocessed_patch)
                break
    
    # we don't want to have to re extract the ids from the patch names.
    id_idx_patches_list = [[key, values] for key, values in zip(ids, idx_patches)]
    return id_idx_patches_list

def combine_preprocessed_patches(patches, model = None):
    # we assume all patches have the same shape and are isomorphic cubes
    image, _ = get_image_mask_from_patch_fp(patches[0])
    patch_size = image.shape[0]
    
    del image

    xs = []
    ys = []
    zs = []
    
    patch_map = {}

    for patch_fp in patches:
        x, y, z = get_patch_coordinates_from_patch_fp(patch_fp)
        x = int(x)
        y = int(y)
        z = int(z)                                    
        
        patch_map[(x, y, z)] = patch_fp
        xs.append(x)
        ys.append(y)
        zs.append(z)
        
    max_x = np.array(xs).max()
    max_y = np.array(ys).max()
    max_z = np.array(zs).max()
    
    # there is no need to reconstruct the image
    reconstructed_mask = np.zeros((patch_size * (max_x + 1),
                            patch_size * (max_y + 1),
                            patch_size * (max_z + 1)))
    
    if model is not None:
        reconstructed_prediction = np.zeros((patch_size * (max_x + 1),
                                patch_size * (max_y + 1),
                                patch_size * (max_z + 1)))
        reconstructed_prediction_before_sigmoid = np.zeros((patch_size * (max_x + 1),
                                patch_size * (max_y + 1),
                                patch_size * (max_z + 1)))
        
    else:
        reconstructed_prediction = None
        reconstructed_prediction_before_sigmoid = None
    
    for i in range(max_x + 1):
        for j in range(max_y + 1):
            for k in range(max_z + 1):
                current_patch = patch_map[(i, j, k)]
                current_image_patch, current_mask_patch = get_image_mask_from_patch_fp(current_patch)

                reconstructed_mask[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size,
                    k * patch_size:(k + 1) * patch_size
                ] = current_mask_patch

                if model is not None:
                    current_image_patch = prepare_image_for_network_input(current_image_patch)
                    with torch.no_grad():
                        current_prediction_patch, current_prediction_patch_before_sigmoid = model(current_image_patch)
                        current_prediction_patch = prepare_image_for_analysis(current_prediction_patch)
                        current_prediction_patch_before_sigmoid = prepare_image_for_analysis(current_prediction_patch_before_sigmoid)
                    
                    reconstructed_prediction[
                        i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size,
                        k * patch_size:(k + 1) * patch_size
                    ] = current_prediction_patch
                    
                    reconstructed_prediction_before_sigmoid[
                        i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size,
                        k * patch_size:(k + 1) * patch_size
                    ] = current_prediction_patch_before_sigmoid

    # Verify if the reconstructed image matches the original dimensions
    # print(reconstructed_mask.shape)
    
    return reconstructed_mask, reconstructed_prediction, reconstructed_prediction_before_sigmoid
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


def get_preprocessed_patches(patches_folder, dummy=False):
    if dummy:
        print("warning: Using dummy data!")
        # they're not even real patches, should fail if used as such
        dummy_fps = np.arange(0, 100)

        return dummy_fps

    patch_fps = list(patches_folder.iterdir())

    patch_fps = [file for file in patch_fps if "ipynb_checkpoints" not in str(file)]

    print("amt of detected patch files: ", len(patch_fps))
    # setting random seed for reproducibility
    random.Random(42).shuffle(patch_fps)

    return patch_fps


def get_image_mask_from_patch_fp(patch_fp, dummy=False):
    # dummy is only to be used for local testing
    if dummy:
        print("warning: Using dummy data!")
        image = np.zeros((128, 128, 128))
        mask = np.zeros((128, 128, 128)).astype(np.bool_)

        return image, mask

    patch = np.load(patch_fp)
    image = patch["image"]
    mask = patch["mask"].astype(np.bool_)

    return image, mask

def get_idx_from_patch_fp(patch_fp):
    file_name = os.path.basename(patch_fp).split('.')[0]
    idx = file_name.split("_")[0]
    return idx

def get_patch_coordinates_from_patch_fp(patch_fp):
    file_name = os.path.basename(patch_fp).split('.')[0]
    x, y, z = file_name.split('_')[-3], file_name.split('_')[-2], file_name.split('_')[-1]
    return x, y, z