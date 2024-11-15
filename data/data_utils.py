# %%
import numpy as np

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

image = np.zeros((275, 512, 512))
padded_img = pad_image(image, patch_size=64)
# %%
