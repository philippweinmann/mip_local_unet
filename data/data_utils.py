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

    # Pad the array with zeros along the first dimension
    padded_image = np.pad(image, ((0, z_pad_length), (0,0), (0, 0)), mode='constant')
    print(f"padding done. Original size: {image.shape}, padded shape: {padded_image.shape}")

    return padded_image
# %%

def divide_3d_image_into_patches(image_3d, block_shape):
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

    image = pad_image(image, patch_size=patch_size)
    mask = pad_image(mask, patch_size=patch_size)

    image_patches = divide_3d_image_into_patches(np.squeeze(image), block_shape=block_shape)
    mask_patches = divide_3d_image_into_patches(np.squeeze(mask), block_shape=block_shape)

    return image_patches, mask_patches