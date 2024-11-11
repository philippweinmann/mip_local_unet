# %%
%load_ext autoreload
%autoreload 2

import copy
import torch
import torch.nn as nn

from data_generation.generate import get_batch, get_image_with_random_shapes

from matplotlib import pyplot as plt
from models.net_visualizations import visualize_model_progress, displayImageMaskTuple

from models.unet import UNet
from models.net_utils import binarize_image_pp
from sklearn.metrics import jaccard_score
# %%
# define which device is used for training

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

torch.set_default_device(device)

print(f"Using {device} device. Every tensor created will be by default on {device}")

# %%
default_model = UNet
model = default_model(in_channels=1, num_classes=1)
model.to(device);

default_image_generation_function = get_image_with_random_shapes

# %%
image, mask = default_image_generation_function()
print("image_shape: ", image.shape)

# %%
displayImageMaskTuple(image, mask)
# %%
print("model prediction at initialization: ")

visualize_model_progress(model, get_image_fct=default_image_generation_function);
# %%
print("----------------TRAINING-------------")
def calculate_jaccard_score(masks, images):
    try:
        masks = masks.detach().cpu().numpy()
        images = images.detach().cpu().numpy()
    except:
        pass

    masks = binarize_image_pp(masks)
    images = binarize_image_pp(images)

    jaccard_scr = jaccard_score(masks.flatten(), images.flatten())
    return jaccard_scr

def train_loop(model, loss_fn, optimizer, batch_size = 10, training_batches = 20):
    model.train()
    
    for _ in range(training_batches):
        images, masks = get_batch(default_image_generation_function, batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        optimizer.zero_grad()

        pred = model(images)
        loss = loss_fn(pred, masks)
        jaccard_score = calculate_jaccard_score(masks, images)

        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        print(f"Train loss: {train_loss:>8f} | Jaccard Score: {jaccard_score:>8f}", end="\r")

# %%
def test_loop(model, loss_fn, batch_size = 10, test_batches = 5):
    model.eval()

    test_loss = 0
    jaccard_score = 0

    for _ in range(test_batches):
        images, masks = get_batch(gen_img_fct = default_image_generation_function, batch_size = batch_size)
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)

        with torch.no_grad():
            pred = model(images)
            test_loss += loss_fn(pred, masks).item()

            jaccard_score += calculate_jaccard_score(masks, images)


    test_loss /= test_batches
    jaccard_score /= test_batches

    print(f"Test loss: {test_loss:>8f}  | Jackard Index: {jaccard_score:>8f} \n", end="\r")

    visualize_model_progress(model, default_image_generation_function)
    
    # to make sure that the plot gets displayed during training
    plt.pause(0.001)

    # returning for patience
    return test_loss


# %%
# resetting the model
model = default_model(in_channels=1, num_classes=1)
model.to(device)

print("model prediction at initialization: ")
visualize_model_progress(model, default_image_generation_function)

# running it
# param initialization for patience
best_loss = float('inf')  
best_model_weights = None  
patience_base_value = 3
patience = patience_base_value

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
print("------INFERENCE--------")
for i in range(10):
    mask, pred = visualize_model_progress(model, get_image_fct=default_image_generation_function)
    print("jaccard score for above image: ", calculate_jaccard_score(mask, pred))

# %%
