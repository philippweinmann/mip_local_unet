from data.data_utils import get_all_patches_with_certain_idx, get_preprocessed_patches
from server_specific.server_utils import get_patients
import training_configuration
import random


def print_logs_to_file(log_line, file_name="train_logs.txt"):
    with open(file_name, "a") as tl:
        tl.write(log_line + "\n")

def get_val_test_indexes():
    patients = get_patients()
    random.Random(42).shuffle(patients)

    val_idxs = []
    test_idxs = []

    counter = 0
    while counter <= training_configuration.AMT_VAL_PATIENTS:
        current_idx = patients[counter].idx
        val_idxs.append(current_idx)
        counter += 1

    counter = 0
    while counter <= training_configuration.AMT_TEST_PATIENTS:
        current_idx = patients[training_configuration.AMT_VAL_PATIENTS + counter].idx
        test_idxs.append(current_idx)
        counter += 1

    print(f"validation patients: {val_idxs}")
    print(f"testing patients: {test_idxs}")

    return val_idxs, test_idxs


def get_train_test_val_patches(patches_folder, dummy):
    preprocessed_patches = get_preprocessed_patches(patches_folder, dummy=dummy)

    if dummy:
        return preprocessed_patches, preprocessed_patches[0:2], preprocessed_patches[2:4]

    val_idxs, test_idxs = get_val_test_indexes()
    test_idxs_patches = get_all_patches_with_certain_idx(test_idxs, preprocessed_patches)
    val_idxs_patches = get_all_patches_with_certain_idx(val_idxs, preprocessed_patches)

    for patch in test_idxs_patches + val_idxs_patches:
        preprocessed_patches.remove(patch)

    return preprocessed_patches, val_idxs_patches, test_idxs_patches
    
