#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from data.data_utils import get_preprocessed_patches
from data.data_utils import get_image_mask_from_patch_fp
from models.net_utils import get_appropriate_dice_weight
from models.net_utils import calculate_learning_rate
import training_configuration

from models.net_utils import save_model, get_best_device, prepare_image_for_network_input, prepare_image_for_analysis

from models.unet3D import UNet3D, DICEBCE
from training_utils import print_logs_to_file

preprocessed_patches = get_preprocessed_patches(training_configuration.PATCHES_FOLDER)

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

logging_frequency = 100

def train_loop(model, loss_fn, optimizer, patch_fps, epoch):
    model.train()
    
    avg_train_loss = 0
    
    amt_of_patches = len(patch_fps)
    for patch_number, patch_fp in enumerate(patch_fps):
        print(f"{patch_number} / {amt_of_patches}", end="\r")
        image_patch, mask_patch = get_image_mask_from_patch_fp(patch_fp)
        
        amt_positive_voxels = np.count_nonzero(mask_patch)
        dynamic_lr = calculate_learning_rate(amt_positive_voxels, epoch)
        
        dynamic_loss_weights = get_appropriate_dice_weight(amt_positive_voxels)
        
        DICEBCE.dice_weight = dynamic_loss_weights[0]
        DICEBCE.bce_weight = dynamic_loss_weights[1]
        
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
            print_logs_to_file(train_log)
            avg_train_loss = 0



# In[ ]:

# resetting the model
model = UNet3D(in_channels=1, num_classes=1)
model.to(device)

# running it
loss_fn = DICEBCE(1,0.5)

# the lr does not matter here, it is set depending on the amt of positive voxels
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 3

# In[ ]:
for epoch in range(epochs):
    train_loop(model, loss_fn, optimizer, preprocessed_patches, epoch)

'''
try:
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        
    print("Done!")
except:
    print("this better have been a keyboard interrupt")
    model.eval()
'''

# In[ ]:

print("------INFERENCE--------")

# In[ ]:

save_model(model)

# In[ ]: