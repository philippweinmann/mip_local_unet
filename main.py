# %%
from generate import get_image
import numpy as np
from matplotlib import pyplot as plt

# %%
image, mask = get_image()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Image')

ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Mask')
# %%