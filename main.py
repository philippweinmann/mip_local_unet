# %%
%load_ext autoreload
%autoreload 2

from generate import get_image, get_batch
from unet import UNet
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
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
def displayImageMaskTuple(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')

image, mask = get_image()
displayImageMaskTuple(image, mask)
# %%
print(image.shape)
image = image[None, None, :, :]
print(image.shape)

# %%
t_image = torch.tensor(image, dtype=torch.float32)
model = UNet(in_channels=1, num_classes=1)
model.to(device)
pred = model(t_image)
# %%
print(pred.shape)
pred_np = pred.detach().cpu().numpy()
pred_np = np.squeeze(pred_np)
print(pred_np.shape)

displayImageMaskTuple(np.squeeze(image), pred_np)
# %%
# train loop
def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 100):
    # Set the model to training mode - important for batch normalization and  dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    
    for batch_number in range(training_batches):
        images, masks = get_batch(batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        # Compute prediction and loss
        pred = model(images)
        loss = loss_fn(pred, masks)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_number % 5 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{batch_number:>5d}/{training_batches:>5d}]")

# %%
# test loop
def test_loop(model, loss_fn, batch_size = 10, test_batches = 20):
    model.eval()
    test_loss = 0

    for batch_number in range(test_batches):
        images, masks = get_batch(batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, mask).item()

    test_loss /= test_batches
    print(f"Test loss: {test_loss:>8f} \n")


# %%
# running it
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, loss_fn, optimizer)
	# testing for each epoch to track the models performance during training.
    test_loop(model, loss_fn)
print("Done!")

# %%
