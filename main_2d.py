# %%
# for archivation purposes and demos.
# %%
# %load_ext autoreload
# %autoreload 2

import copy
import torch
import torch.nn as nn

from data_generation.generate_2d import get_image_with_random_shapes
from data_generation.generate_utils import get_batch
from data_generation.config import cubic_simple_dims

from matplotlib import pyplot as plt
from models.net_utils import calculate_jaccard_score, calculate_dice_score
from models.net_utils import get_best_device
from models.net_visualizations import two_d_visualize_model_progress, display2DImageMaskTuple

from models.unet2D import UNet
# %%
# define which device is used for training

default_image_shape = cubic_simple_dims # only works for 3d

device = get_best_device()

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%

print("setting default functions for two dimensions")
default_model = UNet

default_image_generation_function = get_image_with_random_shapes
default_image_mask_visulization_function = display2DImageMaskTuple
default_model_progress_visualization_function = two_d_visualize_model_progress
default_loss_fn = nn.BCELoss()

model = default_model(in_channels=1, num_classes=1)
model.to(device);
# %%
image, mask = default_image_generation_function()
print("image_shape: ", image.shape)

# %%
default_image_mask_visulization_function(image, mask)
# %%
print("model prediction at initialization: ")

default_model_progress_visualization_function(model, get_image_fct=default_image_generation_function);
# %%
print("----------------TRAINING-------------")

additional_score_tuples = [("jaccard score", calculate_jaccard_score), ("dice score", calculate_dice_score)] # ("hausdorff distance", calculate_hausdorff_distance)

def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 20):
    model.train()
    
    for _ in range(training_batches):
        images, masks = get_batch(default_image_generation_function, batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_fn(pred, masks)
        
        additional_metrics = []
        
        for name, additional_score_function in additional_score_tuples:
            score = additional_score_function(masks, images)
            additional_metrics.append((name, score))

        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        train_log = f"Train loss: {train_loss:>8f}"
        for name, score in additional_metrics:
            train_log = train_log + f" | {name}: {score}"

        print(train_log, end="\r")

# %%
def test_loop(model, loss_fn, batch_size = 10, test_batches = 5):
    model.eval()

    test_loss = 0
    jaccard_score = 0
    dice_score = 0

    for _ in range(test_batches):
        images, masks = get_batch(gen_img_fct = default_image_generation_function, batch_size = batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()

            jaccard_score += calculate_jaccard_score(masks, images)
            dice_score += calculate_dice_score(masks, images)


    test_loss /= test_batches
    jaccard_score /= test_batches
    dice_score /= test_batches

    print(f"Test loss: {test_loss:>8f}  | Jaccard Score: {jaccard_score:>8f} | Dice Score: {dice_score:>8f}\n", end="\r")

    default_model_progress_visualization_function(model, default_image_generation_function)
    
    # to make sure that the plot gets displayed during training
    plt.pause(0.001)

    # returning for patience
    return test_loss


# %%
# resetting the model
model = default_model(in_channels=1, num_classes=1)
model.to(device)

print("model prediction at initialization: ")
default_model_progress_visualization_function(model, default_image_generation_function)

# running it
# param initialization for patience
best_loss = float('inf')  
best_model_weights = None  
patience_base_value = 3
patience = patience_base_value

# loss_fn = nn.BCELoss()
loss_fn = default_loss_fn
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
for i in range(10):
    mask, pred = default_model_progress_visualization_function(model, get_image_fct=default_image_generation_function)
    print("jaccard score for above image: ", calculate_jaccard_score(mask, pred))

# %%
