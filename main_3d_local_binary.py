# for archivation purposes and demos.
# also to debug the model.

# %%
%load_ext autoreload
%autoreload 2

import copy
import torch
import torch.nn as nn
import numpy as np

from data_generation.generate_3d import ImageGenerator
from data_generation.generate_utils import get_batch
from data_generation.config import original_image_shape, cubic_simple_dims

from models.net_utils import get_best_device, save_model, prepare_image_for_network_input, prepare_image_for_analysis
from models.net_visualizations import three_d_visualize_model_progress
from models.cnnBinary import CNN_Binary

# %%
# define which device is used for training

default_image_shape = cubic_simple_dims # only works for 3d

device = get_best_device()

if device == "mps":
    # mps is not supported for 3d
    device = "cpu"

torch.set_default_device(device)
print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
print("setting default functions for three dimensions")
default_model = CNN_Binary
image_generator = ImageGenerator(default_image_shape)

default_image_generation_function = image_generator.get_3DImage
default_model_progress_visualization_function = three_d_visualize_model_progress
default_loss_fn = nn.BCELoss()

model = default_model(in_channels=1)
model.to(device);
# %%
image, mask = default_image_generation_function()
print("image_shape: ", image.shape)

# %%
# model prediction at initialization:
model(prepare_image_for_network_input(image))
# %%
# %%
print("----------------TRAINING-------------")

def coronary_arteries_in_masks(mask_batch):
    present = []
    for mask in mask_batch:
        coronary_arteries_present = np.any(mask)

        present.append(coronary_arteries_present)
    
    present = torch.tensor(present, dtype=torch.float32)
    present = present[:, None]
    return present

def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 20):
    model.train()
    
    for _ in range(training_batches):
        images, masks = get_batch(default_image_generation_function, batch_size)
        images = torch.tensor(images, dtype=torch.float32)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_fn(pred, coronary_arteries_in_masks(masks))

        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        print(f"Train loss: {train_loss:>8f}", end="\r")
# %%
def test_loop(model, loss_fn, batch_size = 10, test_batches = 5):
    model.eval()

    test_loss = 0

    for _ in range(test_batches):
        images, masks = get_batch(gen_img_fct = default_image_generation_function, batch_size = batch_size)
        images = torch.tensor(images, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, coronary_arteries_in_masks(masks)).item()


    test_loss /= test_batches

    print(f"Test loss: {test_loss:>6f}")

    # default_model_progress_visualization_function(model, default_image_generation_function)
    
    # to make sure that the plot gets displayed during training
    # plt.pause(0.001)

    # returning for patience
    return test_loss

# %%
# resetting the model
model = default_model(in_channels=1)
model.to(device)

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
save_model(model)

# %%
print("------INFERENCE--------")