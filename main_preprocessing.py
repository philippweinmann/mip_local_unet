# %%
from pathlib import Path
from server_specific.server_utils import get_patients
import nibabel as nib
import numpy as np
from data.data_utils import pad_image, clip_scans, divide_3d_image_into_patches
from data.preprocessing import resample_image, min_max_normalization
import random

training_voxel_spacing = [0.5,0.5,0.5]
output_dir = Path("/home/tu-philw/group/gecko/pweinmann/mip_local_unet/preprocessed_patches/")
output_dir_centered = Path("/home/tu-philw/group/gecko/pweinmann/mip_local_unet/preprocessed_patches_centered/")

mean_of_means = -125.14 # after voxel spacing fix and clipping

clipping_min = -600
clipping_max = 1000
patch_size = 128

# %%
def pp_and_save_patches_for_patient(patient, output_dir=output_dir, image_only = False):
    image = nib.load(patient.image_fp)
    if not image_only:
        mask = nib.load(patient.label_fp)

    # fix voxel spacing
    original_spacing = image.header.get_zooms()[:3]
    image = resample_image(image.get_fdata(), original_spacing=original_spacing, target_spacing=training_voxel_spacing, is_label=False)
    
    if not image_only:
        mask = resample_image(mask.get_fdata(), original_spacing=original_spacing, target_spacing=training_voxel_spacing, is_label=True)

    # clip image
    image = clip_scans(image, clipping_min, clipping_max)

    # min/max normalize
    image = min_max_normalization(image, clipping_min, clipping_max)
    shape_before_padding = image.shape

    # pad the image and mask as preparation for patching
    image, _ = pad_image(image, patch_size = patch_size)
    
    if not image_only:
        mask, _ = pad_image(mask, patch_size = patch_size)

    # divide the 3d image and mask into patches:
    block_shape = (patch_size, patch_size, patch_size)
    image_patches = divide_3d_image_into_patches(image, block_shape)
    if not image_only:
        mask_patches = divide_3d_image_into_patches(mask, block_shape)

    # let's save them to disk
    image_patch_shape = image_patches.shape
    for x_dim in range(image_patch_shape[0]):
        for y_dim in range(image_patch_shape[1]):
            for z_dim in range(image_patch_shape[2]):
                current_image_patch = image_patches[x_dim, y_dim, z_dim]
                
                # yes I know we should also save x, y and zdim in the patch_dict, but we got something that works, let's not touch it unless necessary.
                file_name = f"{patient.idx}_patch_{x_dim}_{y_dim}_{z_dim}.npz"
                patch_dict = {
                    "image": current_image_patch,
                    "idx": patient.idx,
                    "mask": None if image_only else mask_patches[x_dim, y_dim, z_dim],
                }
                    
                np.savez(output_dir / file_name, **patch_dict)
    
    # returning the shape before padding, so that we can save it 
    # and remove the padding before post processing.
    return shape_before_padding
                
def preprocess_and_save_ccta_scans(patients, amt_patients = None, output_dir=output_dir, image_only = False):
    shapes_before_padding = {}
    
    if amt_patients is None:
        amt_patients = len(patients)

    print(f"preprocessing: {amt_patients} patients")
    counter = 0
    for p_idx, patient in enumerate(patients):
        print(f"processing patient: {p_idx} / {amt_patients}")

        shape_before_padding = pp_and_save_patches_for_patient(patient=patient, output_dir=output_dir, image_only = image_only)
        shapes_before_padding[patient.idx] = shape_before_padding
        counter += 1

        if counter > amt_patients:
            break

    print("Done")
    return shapes_before_padding

# preprocess_and_save_ccta_scans(get_patients(), amt_patients=None, output_dir=output_test_dir)