#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import torch
import torch.nn as nn
import numpy as np
import time

from pathlib import Path
from data.data_utils import get_image_mask_from_patch_fp
from training_utils import get_train_test_val_patches, test_or_validate_model
from models.net_utils import get_appropriate_dice_weight
from models.net_utils import calculate_learning_rate
import training_configuration

from models.net_utils import save_model, get_best_device, prepare_image_for_network_input, prepare_image_for_analysis

from models.unet3D import UNet3D, DICEBCE

local_run = False
if local_run:
    print("warning: running locally")

if not local_run:
    from training_utils import get_val_test_indexes, print_logs_to_file

device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")


print("----------------TRAINING-------------")

patch_size = training_configuration.PATCH_SIZE
block_shape = (patch_size, patch_size, patch_size)
# dice_thresholds = np.array([0.1, 0.25, 0.5])

preprocessed_patches, val_idxs_patches, test_idxs_patches = get_train_test_val_patches(training_configuration.PATCHES_FOLDER, dummy=local_run)

timestr = time.strftime("%Y%m%d-%H%M%S")
training_config = {
    "base_lr": 0.1, # doesn't matter, its calculated dynamically
    "epochs": 10,
    "max_dice_threshold_wf_loss": 22500,
}

print_logs_to_file(str(training_config), file_name = timestr)


def validation_loop(model):
    model.eval()
    avg_overlap_scores, avg_dice_scores, avg_dice_scores_bpp = test_or_validate_model(val_idxs_patches, model)
    # don't forget to set model to training mode again.
    
    return avg_overlap_scores, avg_dice_scores_bpp

logging_frequency = 100
val_logging_frequency = 6000

def train_loop(model, loss_fn, optimizer, patch_fps, epoch, local_run=False, timestr=None):
    model.train()
    
    avg_train_loss = 0
    
    amt_of_patches = len(patch_fps)
    for patch_number, patch_fp in enumerate(patch_fps):
        print(f"{patch_number} / {amt_of_patches}", end="\r")
        image_patch, mask_patch = get_image_mask_from_patch_fp(patch_fp, dummy=local_run)
        
        amt_positive_voxels = np.count_nonzero(mask_patch)
        dynamic_lr = calculate_learning_rate(amt_positive_voxels, epoch)
        
        dynamic_loss_weights = get_appropriate_dice_weight(amt_positive_voxels, max_dice_threshold = training_config["max_dice_threshold_wf_loss"])
        
        loss_fn.dice_weight = dynamic_loss_weights[0]
        loss_fn.bce_weight = dynamic_loss_weights[1]
        
        # Set the learning rate in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = dynamic_lr
        
        image_patch = prepare_image_for_network_input(image_patch)
        mask_patch = prepare_image_for_network_input(mask_patch)

        optimizer.zero_grad()

        patch_pred = model(image_patch)
        loss = loss_fn(patch_pred, mask_patch)

        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()
        
        if (patch_number + 1) % logging_frequency == 0:
            avg_train_loss /= logging_frequency

            train_log = f"Patch number: {patch_number} / {amt_of_patches}, Train loss: {avg_train_loss:>8f}"

            if not local_run:
                print_logs_to_file(train_log, file_name = timestr)
            avg_train_loss = 0
        
        # yes I know it will start at patch 0, that is fine
        if (patch_number + 1) % val_logging_frequency == 0:
            avg_overlap_scores, avg_dice_scores = validation_loop(model)
            validation_log = f"avg_overlap_scores: {avg_overlap_scores}\navg_dice_scores: {avg_dice_scores}"
            print(validation_log)
                  
            print_logs_to_file(validation_log, file_name = timestr)
            
            model.train()

# In[ ]:

# resetting the model
model = UNet3D(in_channels=1, num_classes=1)
model.to(device)

# running it
loss_fn = DICEBCE(1,0.5) # the base weights don't matter, they're set dynamically

# the lr does not matter here, it is set depending on the amt of positive voxels
optimizer = torch.optim.SGD(model.parameters(), lr=training_config["base_lr"])

epochs = training_config["epochs"]

if False:
    for epoch in range(epochs):
        train_loop(model, loss_fn, optimizer, preprocessed_patches, epoch, local_run=local_run, timestr=timestr)
else:
    try:
        for epoch in range(epochs):
            epoch_log = f"epoch: {epoch} / {epochs}"
            print_logs_to_file(epoch_log, file_name = timestr)
            train_loop(model, loss_fn, optimizer, preprocessed_patches, epoch, local_run=local_run, timestr=timestr)
    except:
        print("this better have been a keyboard interrupt")
        model.eval()

# In[ ]:

print("------INFERENCE--------")

# In[ ]:

save_model(model, timestr)

# In[ ]: