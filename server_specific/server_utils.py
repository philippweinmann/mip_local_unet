# %%
from pathlib import Path
import nibabel as nib
from data.preprocessing import preprocess_ccta_scan

training_data_folder = Path("/data/training_data/")
training_files = list(training_data_folder.iterdir())

# %%
class Patient:
    def __init__(self, idx):
        self.idx = idx

    def set_graph_fp(self, graph_fp):
        assert graph_fp.exists(), f"{graph_fp} does not exist"
        self.graph_fp = graph_fp
    
    def set_image_fp(self, image_fp):
        assert image_fp.exists(), f"{image_fp} does not exist"
        self.image_fp = image_fp
    
    def set_label_fp(self, label_fp):
        assert label_fp.exists(), f"{label_fp} does not exist"
        self.label_fp = label_fp

    def get_image_mask_tuple(self):
        image = nib.load(self.image_fp).get_fdata()
        mask = nib.load(self.label_fp).get_fdata()
        
        return image, mask
    
    def get_voxel_spacings(self):
        image = nib.load(self.image_fp)
        original_spacing = image.header.get_zooms()[:3]
        
        return original_spacing
        
    def get_preprocessed_image_mask_tuple(self):
        image, mask = self.get_image_mask_tuple()

        # shift the mean of the image to the average mean
        image = preprocess_ccta_scan(image)

        return image, mask

def get_patients():
    training_files = list(training_data_folder.iterdir())

    # forgotten_ipynb_checkpoints = [file for file in training_files if "ipynb_checkpoints" in str(file)]
    training_files = [file for file in training_files if "ipynb_checkpoints" not in str(file)]

    # print("amt of detected_files: ", len(training_files))

    patients = []

    for training_file in training_files:
        idx = str(training_file).split(".")[0].split("/")[-1]

        # Check if the scan has already been added
        if not any([ccta_scan.idx == idx for ccta_scan in patients]):
            ccta_scan = Patient(idx)
            patients.append(ccta_scan)
        else:
            ccta_scan = [ccta_scan for ccta_scan in patients if ccta_scan.idx == idx][0]


        if "graph" in str(training_file):
            ccta_scan.set_graph_fp(training_file)
        elif "img" in str(training_file):
            ccta_scan.set_image_fp(training_file)
        elif "label" in str(training_file):
            ccta_scan.set_label_fp(training_file)
        elif "ipynb_checkpoints" in str(training_file):
            continue
        else:
            raise ValueError(f"Unknown file type: {training_file}")
    
    # print("amt of patients: ", len(patients))

    return patients



# %%
