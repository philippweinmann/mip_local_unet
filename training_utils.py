from data.data_utils import get_all_patches_with_certain_idx, get_preprocessed_patches, combine_preprocessed_patches
from models.net_utils import calculate_overlap, calculate_dice_scores, binarize_image_pp
from server_specific.server_utils import get_patients
from data_generation.generate_3d import visualize3Dimage, visualize3Dimageandmask
import training_configuration
import random
from skimage.measure import label, regionprops
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from data.visualizations import visualize_3d_matrices, visualize_model_confidence


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

    return val_idxs, test_idxs


def get_train_test_val_patches(patches_folder, dummy = False):
    preprocessed_patches = get_preprocessed_patches(patches_folder, dummy=dummy)

    if dummy:
        return preprocessed_patches, preprocessed_patches[0:2], preprocessed_patches[2:4]

    val_idxs, test_idxs = get_val_test_indexes()
    id_test_idxs_patches = get_all_patches_with_certain_idx(test_idxs, preprocessed_patches)
    id_val_idxs_patches = get_all_patches_with_certain_idx(val_idxs, preprocessed_patches)
    
    # we need only the patches, not the ids
    _, test_idxs_patches = zip(*id_test_idxs_patches)
    _, val_idxs_patches = zip(*id_val_idxs_patches)
    
    # flatten the lists of lists
    test_idxs_patches = [item for sublist in test_idxs_patches for item in sublist]
    val_idxs_patches = [item for sublist in val_idxs_patches for item in sublist]
    
    for non_training_data_patch in test_idxs_patches + val_idxs_patches:
        preprocessed_patches.remove(non_training_data_patch)
            
    print(f"patients for- training: {800 - len(id_val_idxs_patches) - len(id_test_idxs_patches)}, validation: {len(id_val_idxs_patches)}, testing: {len(id_test_idxs_patches)}")
    
    print(f"training patches: {len(preprocessed_patches)}, validation patches: {len(val_idxs_patches)}, test patches: {len(test_idxs_patches)}")
    
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


def test_or_validate_model(id_test_or_val_patches_lists, model, threshold = 0.3, visualize = False, plot_model_confidence = False):
    print(f"selected threshold: {threshold}")

    # pp stands for post processing.
    avg_dice_scores_before_pp = 0
    avg_dice_scores_after_pp = 0
    avg_overlap_scores = 0 # before post processing

    amt_patch_patients = len(id_test_or_val_patches_lists)
    for idx, id_test_or_val_patches_list in enumerate(id_test_or_val_patches_lists):
        # 0. Extract the id and the patches from test_or_val_patches_lists
        patient_id = id_test_or_val_patches_list[0]
        test_or_val_patches_list = id_test_or_val_patches_list[1]
        
        print(f"processing validation or test patient {idx + 1} / {amt_patch_patients}, patient_id: {patient_id}")
        

        
        # 1. Combine the patches to images to be able to get proper scores from them.
        reconstructed_mask, reconstructed_prediction = combine_preprocessed_patches(test_or_val_patches_list, model)
        untouched_reconstructed_prediction = reconstructed_prediction.copy()
        
        # 2. Binarize the prediction
        reconstructed_binarized_prediction_before_preprocessing = binarize_image_pp(reconstructed_prediction, threshold = threshold)
        
        # 3. To be able to visualize the post processing, we copy it before applying post processing.
        # This is in case the post processing is or will be done inplace.
        # The copy will not be touched before plotting it
        reconstructed_binarized_prediction_before_preprocessing_copy = reconstructed_binarized_prediction_before_preprocessing.copy()
        
        # 4. Apply post processing
        reconstructed_prediction_after_pp = post_processing(reconstructed_binarized_prediction_before_preprocessing, threshold = threshold)
        
        # 5. Calculate the dice scores on the image that is preprocessed and the one that is not preprocessed.
        # the thresholds here actually don't matter if we're binarizing before
        patient_dice_scores_before_pp = calculate_dice_scores(reconstructed_mask, reconstructed_binarized_prediction_before_preprocessing, thresholds = [0.5])
        patient_dice_scores_after_pp = calculate_dice_scores(reconstructed_mask, reconstructed_prediction_after_pp, thresholds = [0.5])
        
        overlap_scores = calculate_overlap(reconstructed_mask, reconstructed_binarized_prediction_before_preprocessing, thresholds = [0.5])
        
        # 6. add them to the global ones to provide an average at the end.
        avg_dice_scores_after_pp += patient_dice_scores_after_pp[0]
        avg_dice_scores_before_pp += patient_dice_scores_before_pp[0]
        avg_overlap_scores += overlap_scores[0]

        if visualize:
            matrices = [reconstructed_binarized_prediction_before_preprocessing_copy, reconstructed_prediction_after_pp, reconstructed_mask]
            titles = ["pred, no pp", "pred after pp", "mask"]
            visualize_3d_matrices(matrices, titles, global_title = f"predictions on patient with id: {patient_id}")
            
        if plot_model_confidence:
            visualize_model_confidence(untouched_reconstructed_prediction)
        
        # make some space between the graphs and the scores.
        print("\n\n\n")
        
        print(f"{[f'{score:.4f}' for score in overlap_scores]}: overlap scores")
        print(f"{[f'{score:.4f}' for score in patient_dice_scores_before_pp]}: dice scores before post processing")
        print(f"{[f'{score:.4f}' for score in patient_dice_scores_after_pp]}: dice scores after post processing")
        
        # make space for the next patient
        print("\n\n")
        
    avg_overlap_scores /= amt_patch_patients
    avg_dice_scores_after_pp /= amt_patch_patients
    avg_dice_scores_before_pp /= amt_patch_patients
    
    return avg_overlap_scores, avg_dice_scores_after_pp, avg_dice_scores_before_pp