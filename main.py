# %%
%load_ext autoreload
%autoreload 2

from generate import get_image
from unet import UNet
import numpy as np
from matplotlib import pyplot as plt
import torch
# %%
image, mask = get_image()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Image')

ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Mask')
# %%
print(image.shape)
image = image[None, None, :, :]
print(image.shape)

# %%
t_image = torch.tensor(image).float()
model = UNet(in_channels=1, num_classes=1)
# %%
model(t_image)
# %%
