# %%
%load_ext autoreload
%autoreload 2
from data.visualizations import create_2Dimagegrid
import numpy as np

# %%
image = np.random.rand(64, 64)
images = [image, image]
titles = ["title 1", "title 2"]

create_2Dimagegrid(images, titles, global_title="global title")
# %%
