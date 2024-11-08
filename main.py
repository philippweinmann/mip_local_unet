# %%
%load_ext autoreload
%autoreload 2

import sys
from generate import get_image_with_stripes, get_batch, get_image_with_random_shapes, get_image_with_random_shape_small_mask
from unet import UNet
from simple_unet import SimpleUNet
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from net_utils import get_weighted_bce_loss, iou, dice_loss, visualize_model_parameters, prepare_image_for_network_input
import copy
# %%
# define which device is used for training

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

# %%
default_model = UNet
# default_image_generation_function = get_image_with_random_shape_small_mask
default_image_generation_function = get_image_with_random_shapes

# model.to(device) # do this later if the model is defined later.
torch.set_default_device(device)

print(f"Using {device} device. Every tensor created will be by default on {device}")
# %%
def displayImageMaskTuple(image, mask, predicted_mask = None):
    if predicted_mask is not None:
        amt_subplots = 3
    else:
        amt_subplots = 2
    
    fig, ax = plt.subplots(1, amt_subplots, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')

    if predicted_mask is not None:
        ax[2].imshow(predicted_mask, cmap='gray')
        ax[2].set_title('Predicted Mask')

# image, mask = get_image_with_stripes()
image, mask = default_image_generation_function()
displayImageMaskTuple(image, mask)
# %%
print(image.shape)
image = image[None, None, :, :]
print(image.shape)

# %%
model = default_model(in_channels=1, num_classes=1)
model.to(device);
# %%
def visualize_model_progress(model, get_image_fct = default_image_generation_function):
    original_img, original_mask = get_image_fct()
    image = original_img[None, None, :, :]

    t_image = torch.tensor(image, dtype=torch.float32)
    pred = model(t_image)
    pred_np = pred.detach().cpu().numpy()
    pred_np = np.squeeze(pred_np)

    displayImageMaskTuple(original_img, original_mask, pred_np)


visualize_model_progress(model, get_image_fct=default_image_generation_function)
# %%
# train loop
def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 20):
    # Set the model to training mode - important for batch normalization and  dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    
    for batch_number in range(training_batches):
        images, masks = get_batch(default_image_generation_function, batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(images)
        loss = loss_fn(pred, masks)
        jackard_index = iou(masks.detach().cpu().numpy(), pred.detach().cpu().numpy())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Let#s visualize the gradients to detect potential issues here
        if batch_number % 10 == 0:
            # visualize_model_parameters(model, batch_number)
            pass

        if batch_number % 1 == 0:
            train_loss = loss.item()

            train_log = f"Train loss: {train_loss:>8f}  | Jackard Index: {jackard_index:>8f} \n"
            sys.stdout.write('\r' + train_log)
            sys.stdout.flush()

# %%
# test loop
def test_loop(model, loss_fn, batch_size = 10, test_batches = 5):
    model.eval()
    test_loss = 0
    jackard_index = 0

    for batch_number in range(test_batches):
        images, masks = get_batch(gen_img_fct = default_image_generation_function, batch_size = batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()

            jackard_index += iou(masks.detach().cpu().numpy(), pred.detach().cpu().numpy())

    test_loss /= test_batches
    jackard_index /= test_batches

    test_log = f"Test loss: {test_loss:>8f}  | Jackard Index: {jackard_index:>8f} \n"
    sys.stdout.write('\r' + test_log)
    sys.stdout.flush()

    visualize_model_progress(model, default_image_generation_function)
    # to make sure that the plot gets displayed during training
    plt.pause(0.001)

    return test_loss


# %%
# resetting the model
model = default_model(in_channels=1, num_classes=1)
model.to(device)

visualize_model_progress(model, default_image_generation_function)

# running it
# param initialization for patience
best_loss = float('inf')  
best_model_weights = None  
patience_base_value = 3
patience = patience_base_value

# loss_fn = get_weighted_bce_loss
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 200

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
for i in range(10):
    visualize_model_progress(model, get_image_fct=default_image_generation_function)
# %%
image, mask = default_image_generation_function()
t_image = prepare_image_for_network_input(image)

pred = model(t_image)
np_pred = np.squeeze(pred.cpu().detach().numpy())
plt.imshow(np_pred, cmap='gray')
# %%
np_pred.max()
# %%
image.max()
# %%
