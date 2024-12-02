from data.data_utils import get_patients
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
    
