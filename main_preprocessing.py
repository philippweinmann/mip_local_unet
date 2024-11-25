# %%
from pathlib import Path
from server_specific.server_utils import get_patients
import nibabel as nib
import numpy as np
from data.data_utils import pad_image, resample_image, clip_scans, min_max_normalize, divide_3d_image_into_patches

training_voxel_spacing = [0.5,0.5,0.5]
output_dir = Path("/home/tu-philw/group/gecko/pweinmann/mip_local_unet/preprocessed_patches/")
clipping_min = -600
clipping_max = 1000
patch_size = 128

# %%
def save_patches_for_patient(patient):
    image = nib.load(patient.image_fp)
    mask = nib.load(patient.label_fp)

    # fix voxel spacing
    original_spacing = image.header.get_zooms()[:3]
    image = resample_image(image.get_fdata(), original_spacing=original_spacing, target_spacing=training_voxel_spacing)
    mask = resample_image(mask.get_fdata(), original_spacing=original_spacing, target_spacing=training_voxel_spacing)

    # clip image
    image = clip_scans(image, clipping_min, clipping_max)

    # min/max normalize
    image = min_max_normalize(image, clipping_min, clipping_max)

    # pad the image and mask as preparation for patching
    image, _ = pad_image(image, patch_size = patch_size)
    mask, _ = pad_image(mask, patch_size = patch_size)

    # divide the 3d image and mask into patches:
    block_shape = (patch_size, patch_size, patch_size)
    image_patches = divide_3d_image_into_patches(image, block_shape)
    mask_patches = divide_3d_image_into_patches(mask, block_shape)

    # let's save them to disk
    image_patch_shape = image_patches.shape
    for x_dim in range(image_patch_shape[0]):
        for y_dim in range(image_patch_shape[1]):
            for z_dim in range(image_patch_shape[2]):
                current_image_patch = image_patches[x_dim, y_dim, z_dim]
                current_mask_patch = mask_patches[x_dim, y_dim, z_dim]

                np.savez(output_dir / f"{patient.idx}_image_and_mask_patch_{x_dim}_{y_dim}_{z_dim}.npz", image = current_image_patch, mask = current_mask_patch)

def preprocess_and_save_ccta_scans(patients, amt_patients = None):
    if amt_patients is None:
        amt_patients = len(patients)

    print(f"preprocessing: {amt_patients} patients")
    counter = 0
    for p_idx, patient in enumerate(patients):
        print(f"processing patient: {p_idx} / {amt_patients}")

        save_patches_for_patient(patient=patient)
        counter += 1

        if counter > amt_patients:
            break

    print("Done")

preprocess_and_save_ccta_scans(patients = get_patients())