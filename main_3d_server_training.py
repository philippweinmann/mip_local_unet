#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import torch
import torch.nn as nn
import numpy as np
import random

from pathlib import Path
import training_configuration
from data_generation.generate_3d import ImageGenerator
from data_generation.generate_utils import get_batch
from data_generation.config import original_image_shape, cubic_simple_dims

from models.net_utils import calculate_jaccard_score, calculate_dice_scores, save_model
from models.net_utils import get_best_device, prepare_image_for_network_input, prepare_image_for_analysis
from models.net_visualizations import three_d_visualize_model_progress, display3DImageMaskTuple

from models.unet3D import UNet3D, dice_bce_loss, DICEBCE
from data.data_utils import pad_image, divide_3d_image_into_patches, get_padded_patches

from sklearn.model_selection import train_test_split
from server_specific.server_utils import get_patients


# In[ ]:


patches_folder = Path("/home/tu-philw/group/gecko/pweinmann/mip_local_unet/preprocessed_patches/")
def get_preprocessed_patches():
    patch_fps = list(patches_folder.iterdir())

    patch_fps = [file for file in patch_fps if "ipynb_checkpoints" not in str(file)]

    print("amt of detected patch files: ", len(patch_fps))
    random.shuffle(patch_fps)
    
    return patch_fps

def get_image_mask_from_patch_fp(patch_fp):
    patch = np.load(patch_fp)
    image = patch["image"]
    mask = patch["mask"].astype(np.bool_)

    return image, mask


# In[ ]:


preprocessed_patches = get_preprocessed_patches()


# In[ ]:


# define which device is used for training
device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")


# In[ ]:


max_dice_threshold = 20000
total_weight = 1.5
min_bce_weight = 0.2

def get_appropriate_dice_weight(amt_positive_voxels):
    bce_weight = (-(total_weight - min_bce_weight)/max_dice_threshold) * amt_positive_voxels + 1.5
    dice_weight = total_weight - bce_weight
    
    return dice_weight, bce_weight


# In[ ]:


pos_voxel_threshold = 7000
max_lr_threshold = 0.1

def calculate_learning_rate(amt_positive_voxels, epoch):
    # make the dice loss an exponential function. 0.0001 if there are no pos voxels, 0.1 if above pos_voxel_threshold
    lr = 10 ** -(epoch) * 0.00001 * np.exp(amt_positive_voxels * (3*np.log(10)/pos_voxel_threshold))
    lr = min(lr, max_lr_threshold)
    
    # print(f"learning rate: {lr}, amt_positive_voxels: {amt_positive_voxels}")
    return lr


# In[ ]:


print("----------------TRAINING-------------")

patch_size = training_configuration.PATCH_SIZE
block_shape = (patch_size, patch_size, patch_size)
dice_thresholds = np.array([0.1, 0.25, 0.5])

logging_frequency = 48

def train_loop(model, loss_fn, optimizer, patch_fps, epoch):
    model.train()
    
    avg_train_loss = 0
    avg_dice_scores = np.zeros(dice_thresholds.shape)
    
    processed_patch_counter = 0
    
    combined_mask = []
    combined_pred = []
    
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

        train_loss = loss.item()
        avg_train_loss += train_loss

        # list of tuples with threshold, score
        mask_patch = prepare_image_for_analysis(mask_patch)
        patch_pred = prepare_image_for_analysis(patch_pred)
        patch_pred.astype(np.float16)
        
        combined_mask.append(mask_patch)
        combined_pred.append(patch_pred)
        
        if (patch_number + 1) % logging_frequency == 0:
            avg_train_loss /= logging_frequency
            
            combined_mask_array = np.concatenate(combined_mask, axis=0)
            combined_pred_array = np.concatenate(combined_pred, axis=0)
            
            dice_scores = calculate_dice_scores(combined_mask_array, combined_pred_array, thresholds=dice_thresholds)
            
            formatted_scores = ', '.join([f'({t:.2f}, {s:.6f})' for t, s in zip(dice_thresholds, dice_scores)])
            train_log = f"Patch number: {patch_number} / {amt_of_patches}, Train loss: {avg_train_loss:>8f}, Dice Scores: {formatted_scores}"
            print(train_log)
            
            # reset the scores again
            combined_mask = []
            combined_pred = []
            avg_train_loss = 0
            avg_dice_scores.fill(0)


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


try:
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loop(model, loss_fn, optimizer, preprocessed_patches, epoch)
    print("Done!")
except:
    print("this better have been a keyboard interrupt")
    model.eval()


# In[ ]:


print("------INFERENCE--------")


# In[ ]:


debugging = False

if not debugging:
    save_model(model)


# In[ ]:




