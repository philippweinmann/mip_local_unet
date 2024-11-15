# This file needs to be able to run on the main server. 
# Do not put anything in here that doesn't run on the main server.

# no clutter, single purpose.
# only meant to be run on data that has the same shape as the medical ccta scans.
# %%
%load_ext autoreload
%autoreload 2

import copy
import torch
import numpy as np

from data_generation.generate_3d import ImageGenerator
from data_generation.generate_utils import get_batch
from data_generation.config import original_image_shape, cubic_simple_dims

from matplotlib import pyplot as plt
from models.net_utils import calculate_jaccard_score, calculate_dice_score
from models.net_utils import get_best_device, prepare_image_for_network_input, prepare_image_for_analysis
from models.net_visualizations import three_d_visualize_model_progress, display3DImageMaskTuple

from models.unet3D import UNet3D, dice_bce_loss
from data.data_utils import pad_image, divide_3d_image_into_patches, get_padded_patches
# %%
# define which device is used for training
# todo replace with original image shape
default_image_shape = original_image_shape # only works for 3d

device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
print("setting default functions for three dimensions")
image_generator = ImageGenerator(default_image_shape)

default_image_mask_visulization_function = display3DImageMaskTuple
default_model_progress_visualization_function = three_d_visualize_model_progress

model = UNet3D(in_channels=1, num_classes=1)
model.to(device);
# %%
image, mask = image_generator.get_3DImage()
print("image_shape: ", image.shape)

# %%
# default_image_mask_visulization_function(image, mask)
# %%
# print("model prediction at initialization: ")

# default_model_progress_visualization_function(model, get_image_fct=image_generator.get_3DImage);
# %%
print("----------------TRAINING-------------")

additional_score_tuples = [("jaccard score", calculate_jaccard_score), ("dice score", calculate_dice_score)] # ("hausdorff distance", calculate_hausdorff_distance)

padded_shape = (320, 512, 512)
patch_size = 64
block_shape = (patch_size, patch_size, patch_size)

def train_loop(model, loss_fn, optimizer):
    model.train()

    image, mask = image_generator.get_3DImage()

    image_patches, mask_patches = get_padded_patches(image, mask, patch_size)

    patch_shape = image_patches.shape
    amt_of_image_patches = patch_shape[0] * patch_shape[1] * patch_shape[2]
    # reconstructed_prediction_mask = np.zeros((padded_shape))
    
    patch_counter = 0
    for i in range(patch_shape[0]):
        for j in range(patch_shape[1]):
            for k in range(patch_shape[2]):
                patch_counter += 1
                current_image_patch = image_patches[i, j, k]
                current_mask_patch = mask_patches[i, j, k]

                current_image_patch = prepare_image_for_network_input(current_image_patch)
                current_mask_patch = prepare_image_for_network_input(current_mask_patch)


                optimizer.zero_grad()

                current_prediction_patch = model(current_image_patch)
                loss = loss_fn(current_prediction_patch, current_mask_patch)
                
                additional_metrics = []
                
                for name, additional_score_function in additional_score_tuples:
                    score = additional_score_function(current_mask_patch, current_image_patch)
                    additional_metrics.append((name, score))

                loss.backward()
                optimizer.step()

                train_loss = loss.item()

                train_log = f"Patch number: {patch_counter} / {amt_of_image_patches}, Train loss: {train_loss:>8f}"
                for name, score in additional_metrics:
                    train_log = train_log + f" | {name}: {score}"

                print(train_log, end="\r")

                # only useful for testing?
                '''current_prediction_patch = prepare_image_for_analysis(current_prediction_patch)
                reconstructed_prediction_mask[
                    i * block_shape[0]:(i + 1) * block_shape[0],
                    j * block_shape[1]:(j + 1) * block_shape[1],
                    k * block_shape[2]:(k + 1) * block_shape[2]
                ] = current_prediction_patch'''

# %%
def test_loop(model, loss_fn):
    model.eval()

    test_loss = 0
    jaccard_score = 0
    dice_score = 0

    image, mask = image_generator.get_3DImage()
    image_patches, mask_patches = get_padded_patches(image, mask, patch_size)

    patch_shape = image_patches.shape
    amt_of_image_patches = patch_shape[0] * patch_shape[1] * patch_shape[2]
    # reconstructed_prediction_mask = np.zeros((padded_shape))
    
    patch_counter = 0
    for i in range(patch_shape[0]):
        for j in range(patch_shape[1]):
            for k in range(patch_shape[2]):
                patch_counter += 1
                current_image_patch = image_patches[i, j, k]
                current_mask_patch = mask_patches[i, j, k]

                current_image_patch = prepare_image_for_network_input(current_image_patch)
                current_mask_patch = prepare_image_for_network_input(current_mask_patch)

                with torch.no_grad():
                    current_prediction_patch = model(current_image_patch)
                    patch_loss = loss_fn(current_prediction_patch, current_mask_patch).item()
                    test_loss += patch_loss

                    patch_jaccard_score = calculate_jaccard_score(current_mask_patch, current_prediction_patch)
                    patch_dice_score = calculate_dice_score(current_mask_patch, current_prediction_patch)

                    jaccard_score += patch_jaccard_score
                    dice_score += patch_dice_score

                    print(f"Patch number: {patch_counter} / {amt_of_image_patches}, Test loss: {patch_loss:>8f}  | Jaccard Score: {patch_jaccard_score:>8f} | Dice Score: {patch_dice_score:>8f}\n", end="\r")

    test_loss /= amt_of_image_patches
    jaccard_score /= amt_of_image_patches
    dice_score /= amt_of_image_patches

    print(f"Patch number: {patch_counter} / {amt_of_image_patches}, average test loss: {test_loss:>8f}  | average Jaccard Score: {jaccard_score:>8f} | average Dice Score: {dice_score:>8f}\n", end="\r")

    # default_model_progress_visualization_function(model, image_generator.get_3DImage)
    
    # to make sure that the plot gets displayed during training
    # plt.pause(0.001)

    # returning for patience
    return test_loss
# %%
# resetting the model
model = UNet3D(in_channels=1, num_classes=1)
model.to(device)

# print("model prediction at initialization: ")
# default_model_progress_visualization_function(model, image_generator.get_3DImage)

# running it
# param initialization for patience
best_loss = float('inf')  
best_model_weights = None  
patience_base_value = 3
patience = patience_base_value

# loss_fn = nn.BCELoss()
loss_fn = dice_bce_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 200

# %%
try:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, loss_fn, optimizer)
        # testing for each epoch to track the models performance during training.
        
        test_loss = test_loop(model, loss_fn)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = patience_base_value
        else:
            patience -= 1
            if patience == 0:
                break
        print("patience: ", patience)
    print("Done!")
except KeyboardInterrupt:
    print("training interrupted by the user")
    model.eval()

# %%
print("------INFERENCE--------")

'''
for i in range(10):
    mask, pred = default_model_progress_visualization_function(model, get_image_fct=image_generator.get_3DImage)
    print("jaccard score for above image: ", calculate_jaccard_score(mask, pred))
'''
# %%
