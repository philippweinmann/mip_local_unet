from data.data_utils import get_all_patches_with_certain_idx, get_preprocessed_patches, combine_preprocessed_patches
from models.net_utils import calculate_overlap, calculate_dice_scores
from server_specific.server_utils import get_patients
from data_generation.generate_3d import visualize3Dimage
import training_configuration
import random


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

def test_or_validate_model(train_or_val_patches_lists, model, visualize = False):
    avg_dice_scores = 0
    avg_overlap_scores = 0
    
    amt_patch_patients = len(train_or_val_patches_lists)
    for idx, train_or_val_patches_list in enumerate(train_or_val_patches_lists):
        print(f"processing validation or test patient {idx} / {amt_patch_patients}")
        
        reconstructed_mask, reconstructed_prediction = combine_preprocessed_patches(train_or_val_patches_list, model)

        dice_scores = calculate_dice_scores(reconstructed_mask, reconstructed_prediction, thresholds = [0.5])
        overlap_scores = calculate_overlap(reconstructed_mask, reconstructed_prediction, thresholds = [0.5])

        avg_dice_scores += dice_scores[0]
        avg_overlap_scores += overlap_scores[0]
        
        print(f"dice scores: {dice_scores}")
        print(f"overlap scores: {overlap_scores}")
        
        if visualize:
            visualize3Dimage(reconstructed_mask)
            visualize3Dimage(reconstructed_prediction)
    
    avg_overlap_scores /= amt_patch_patients
    avg_dice_scores /= amt_patch_patients
    
    return avg_overlap_scores, avg_dice_scores