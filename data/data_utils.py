# %%
import numpy as np
from skimage.util import view_as_blocks

def pad_image(image, patch_size):
    # we expect the z_dimension to need padding
    z_dim = image.shape[0]

    rest = z_dim % patch_size
    if rest == 0:
        print(f"no padding required. Original image shape: {image.shape}")
        return image

    # we need to add that many slices
    z_pad_length = patch_size - rest

    # Pad the array with zeros along the first dimension, the zeroes are added at the end
    padded_image = np.pad(image, ((0, z_pad_length), (0,0), (0, 0)), mode='constant')
    # print(f"padding done. Original size: {image.shape}, padded shape: {padded_image.shape}")

    assert np.all(padded_image[-z_pad_length:0] == 0)

    return padded_image, z_pad_length
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

def calculate_voxel_intensities_of_the_masked_area(images, masks):
    bin_edges = np.arange(-2000, 2000, 100)
    total_counts = np.zeros(len(bin_edges) - 1)

    amt_images = len(images)
    for image_idx, (image, mask) in enumerate(zip(images, masks)):
        print(f"processing image {image_idx + 1} / {amt_images}")
        masked_image = image[mask > 0]
        counts, _ = np.histogram(masked_image, bin_edges)
        total_counts += counts
        
    return total_counts, bin_edges

    


