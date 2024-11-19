ccta_scans_dims = 512
ccta_scans_slices = 275
mean_voxel_intensity = -186

original_image_shape = (ccta_scans_slices, ccta_scans_dims, ccta_scans_dims) # z , y , x

# for faster execution and debugging
cubic_simple_dims = (original_image_shape[1] // 16, original_image_shape[1] // 16, original_image_shape[1] // 16)