import numpy as np

def get_batch(gen_img_fct, batch_size: int):
    images = []
    masks = []
    for _ in range(batch_size):
        image, mask = gen_img_fct()
        images.append(image[None, :, :])
        masks.append(mask[None, :, :])

    return np.array(images), np.array(masks)