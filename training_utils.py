from data.data_utils import get_all_patches_with_certain_idx, get_preprocessed_patches, combine_preprocessed_patches
from models.net_utils import calculate_overlap, calculate_dice_scores, binarize_image_pp
from server_specific.server_utils import get_patients
from data_generation.generate_3d import visualize3Dimage, visualize3Dimageandmask
import training_configuration
import random
from skimage.measure import label, regionprops
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def print_logs_to_file(log_line, file_name=None):
    file_path = training_configuration.TRAINING_LOGS_FOLDER / (file_name + ".txt")
    with open(file_path, "a") as tl:
        tl.write(log_line + "\n")

def get_val_test_indexes():
    patients = get_patients()
    random.Random(42).shuffle(patients)

    val_idxs = []
    test_idxs = []

    counter = 0
    while counter < training_configuration.AMT_VAL_PATIENTS:
        current_idx = patients[counter].idx
        val_idxs.append(current_idx)
        counter += 1

    counter = 0
    while counter < training_configuration.AMT_TEST_PATIENTS:
        current_idx = patients[training_configuration.AMT_VAL_PATIENTS + counter + 1].idx
        test_idxs.append(current_idx)
        counter += 1

    print(f"validation patients: {val_idxs}")
    print(f"testing patients: {test_idxs}")

    return val_idxs, test_idxs


def get_train_test_val_patches(patches_folder, dummy = False):
    preprocessed_patches = get_preprocessed_patches(patches_folder, dummy=dummy)

    if dummy:
        return preprocessed_patches, preprocessed_patches[0:2], preprocessed_patches[2:4]

    val_idxs, test_idxs = get_val_test_indexes()
    test_idxs_patches = get_all_patches_with_certain_idx(test_idxs, preprocessed_patches)
    val_idxs_patches = get_all_patches_with_certain_idx(val_idxs, preprocessed_patches)

    for idx_patches in test_idxs_patches + val_idxs_patches:
        for patch in idx_patches:
            # print(patch)
            preprocessed_patches.remove(patch)

    return preprocessed_patches, val_idxs_patches, test_idxs_patches

def post_processing(prediction, threshold, distance_threshold_ratio = 0.5):
    # let's binarize the prediction
    prediction = binarize_image_pp(prediction, threshold = threshold)
    
    # use the center of the image to determine if we keep the element or not.
    image_shape = prediction.shape
    center_z, center_y, center_x = image_shape[0] / 2, image_shape[1] / 2, image_shape[2] / 2
    
    # Calculate the maximum possible distance from center to a corner in 3D
    max_distance = np.sqrt(center_z**2 + center_y**2 + center_x**2)
    
    # Define the distance threshold
    distance_threshold = max_distance * distance_threshold_ratio

    # Label connected components
    labeled_array, num_features = label(prediction, return_num=True)

    # Use a histogram to efficiently calculate the sizes of each labeled component
    component_sizes = np.bincount(labeled_array.ravel())

    # Exclude the background component (label 0)
    component_sizes[0] = 0

    # Get the two largest component labels
    # largest_labels = np.argsort(component_sizes)
    regions = regionprops(labeled_array)
    
    valid_labels = []
    for region in regions:
        centroid_z, centroid_y, centroid_x = region.centroid
        distance = np.sqrt(
            (centroid_z - center_z) ** 2 +
            (centroid_y - center_y) ** 2 +
            (centroid_x - center_x) ** 2
        )
        
        # print("detected distance: ", distance)
    
    # print(f"distance threshold: {distance_threshold}")
    # Filter based on distance threshold
        if distance <= distance_threshold:
                valid_labels.append((region.label, region.area))
    
    # If no regions are valid after distance filtering, return empty mask
    if not valid_labels:
        return np.zeros_like(prediction, dtype=np.uint8)
    
    # Sort the valid labels based on area (volume) in descending order
    valid_labels_sorted = sorted(valid_labels, key=lambda x: x[1], reverse=True)
    
    # Select the top 'num_largest' labels
    top_labels = [label for label, size in valid_labels_sorted[:2]]
    

    # Create a mask for the two largest components
    result_array = np.isin(labeled_array, top_labels).astype(np.uint8)

    return result_array


def test_or_validate_model(train_or_val_patches_lists, model, threshold = 0.3, visualize = False):
    
    print(f"selected threshold: {threshold}")

    avg_dice_scores = 0
    avg_dice_scores_b_pp = 0
    avg_overlap_scores = 0

    amt_patch_patients = len(train_or_val_patches_lists)
    for idx, train_or_val_patches_list in enumerate(train_or_val_patches_lists):
        print(f"processing validation or test patient {idx + 1} / {amt_patch_patients}")

        reconstructed_mask, reconstructed_prediction = combine_preprocessed_patches(train_or_val_patches_list, model)
        reconstructed_prediction_without_preprocessing = reconstructed_prediction.copy()
        reconstructed_prediction = post_processing(reconstructed_prediction, threshold = threshold)
        
        # the thresholds here actually don't matter if we're binarizing before
        dice_scores = calculate_dice_scores(reconstructed_mask, reconstructed_prediction, thresholds = [0.5])
        dice_scores_bpp = calculate_dice_scores(reconstructed_mask, reconstructed_prediction_without_preprocessing, thresholds = [0.5])
        overlap_scores = calculate_overlap(reconstructed_mask, reconstructed_prediction, thresholds = [0.5])

        avg_dice_scores += dice_scores[0]
        avg_dice_scores_b_pp += dice_scores_bpp[0]
        avg_overlap_scores += overlap_scores[0]
    
        print(f"dice scores before post processing: {dice_scores_bpp}")
        print(f"dice scores after post processing: {dice_scores}")
        print(f"overlap scores: {overlap_scores}")

        if visualize:
            reconstructed_prediction = binarize_image_pp(reconstructed_prediction)
            reconstructed_prediction_without_preprocessing = binarize_image_pp(reconstructed_prediction_without_preprocessing)
            visualize3Dimageandmask(reconstructed_prediction, reconstructed_mask)
            visualize3Dimage(reconstructed_prediction_without_preprocessing)

    avg_overlap_scores /= amt_patch_patients
    avg_dice_scores /= amt_patch_patients
    avg_dice_scores_b_pp /= amt_patch_patients
    
    print(f"average dice scores before pp: {avg_dice_scores_b_pp}")
    return avg_overlap_scores, avg_dice_scores, avg_dice_scores_b_pp