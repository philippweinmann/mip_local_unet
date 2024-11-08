# %%
%load_ext autoreload
%autoreload 2

from generate import get_image_with_stripes, get_batch, get_image_with_random_shapes
from unet import UNet
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from net_utils import get_weighted_bce_loss, iou, dice_loss
# %%
# define which device is used for training

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

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
image, mask = get_image_with_random_shapes()
displayImageMaskTuple(image, mask)
# %%
print(image.shape)
image = image[None, None, :, :]
print(image.shape)

# %%
model = UNet(in_channels=1, num_classes=1)
model.to(device);
# %%
def visualize_model_progress(model, get_image_fct):
    original_img, original_mask = get_image_fct()
    image = original_img[None, None, :, :]

    t_image = torch.tensor(image, dtype=torch.float32)
    pred = model(t_image)
    pred_np = pred.detach().cpu().numpy()
    pred_np = np.squeeze(pred_np)

    displayImageMaskTuple(original_img, original_mask, pred_np)


visualize_model_progress(model, get_image_fct=get_image_with_random_shapes)
# %%
# train loop
def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 20):
    # Set the model to training mode - important for batch normalization and  dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    
    for batch_number in range(training_batches):
        images, masks = get_batch(get_image_with_random_shapes, batch_size)
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
        
        if batch_number % 1 == 0:
            loss = loss.item()
            print(f"Jackard Index: {jackard_index:>8f} \n")
            print(f"loss: {loss:>7f}  [{batch_number:>5d}/{training_batches:>5d}]")

# %%
# test loop
def test_loop(model, loss_fn, batch_size = 10, test_batches = 5):
    model.eval()
    test_loss = 0
    jackard_index = 0

    for batch_number in range(test_batches):
        images, masks = get_batch(gen_img_fct = get_image_with_random_shapes, batch_size = batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()

            jackard_index += iou(masks.detach().cpu().numpy(), pred.detach().cpu().numpy())

    test_loss /= test_batches
    jackard_index /= test_batches
    print(f"Test loss: {test_loss:>8f} \n")
    print(f"Jackard Index: {jackard_index:>8f} \n")

    visualize_model_progress(model, get_image_with_random_shapes)
    # to make sure that the plot gets displayed during training
    plt.pause(0.001)


# %%
# resetting the model
model = UNet(in_channels=1, num_classes=1)
model.to(device)

visualize_model_progress(model, get_image_with_random_shapes)

# running it
# loss_fn = get_weighted_bce_loss
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 20

try:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, loss_fn, optimizer)
        # testing for each epoch to track the models performance during training.
        
        test_loop(model, loss_fn)
    print("Done!")
except KeyboardInterrupt:
    
    print("training interrupted by the user")

# %%
visualize_model_progress(model, get_image_fct=get_image_with_random_shapes)


# %%
