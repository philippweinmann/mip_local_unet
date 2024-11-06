# %%
# let's generate the data. We want simple masks, the model should detect black data.

import numpy as np
# %%
x_dim = 256
y_dim = 256 # 256 x 256 images

img_shape = (x_dim, y_dim)
# %%
def get_image():
    '''ignore the stripes at the top and bottom of the image'''

    rand_int_beg = np.random.randint(0, 40 - 5)
    rand_int_end = np.random.randint(y_dim - 40, y_dim - 5)

    rand_int = np.random.randint(0, 10)
    image = np.zeros(img_shape)
    image[rand_int_beg:rand_int_beg + 5] = 1
    image[rand_int_end:rand_int_end + 5] = 1

    mask = np.zeros(img_shape)

    for _ in range(rand_int):
        rand_int = np.random.randint(40, y_dim - 40 - 5)
        image[rand_int:rand_int + 5] = 1
        mask[rand_int:rand_int + 5] = 1

    return image, mask

def get_batch(batch_size: int):
    images = []
    masks = []
    for _ in range(batch_size):
        image, mask = get_image()
        images.append(image[None,:,:])
        masks.append(mask[None,:,:])

    return np.array(images), np.array(masks)