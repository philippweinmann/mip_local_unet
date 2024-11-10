# %%
# let's generate the data. We want simple masks, the model should detect black data.

import numpy as np
import cv2

from config import ccta_scans_dims, ccta_scans_slices

# %%
simplified_dims = ccta_scans_dims // 2# for faster execution
x_dim = simplified_dims
y_dim = simplified_dims  # 256 x 256 images

img_shape = (y_dim, x_dim)


# %%
# This doesn't work, because the Unet does not have any spatial information. 
# It won't be able to know where the stripes are.
def get_image_with_stripes():
    """the model should ignore the stripes at the top and bottom of the image"""

    rand_int_beg = np.random.randint(0, 40 - 5)
    rand_int_end = np.random.randint(y_dim - 40, y_dim - 5)

    rand_int = np.random.randint(0, 10)
    image = np.zeros(img_shape)
    image[rand_int_beg : rand_int_beg + 5] = 1
    image[rand_int_end : rand_int_end + 5] = 1

    mask = np.zeros(img_shape)

    for _ in range(rand_int):
        rand_int = np.random.randint(40, y_dim - 40 - 5)
        image[rand_int : rand_int + 5] = 1
        mask[rand_int : rand_int + 5] = 1

    # note that the foreground object (in the mask)
    # need to be classified as a 1.
    # if not, then some loss functions won't work
    # as expected (dice loss)
    return image, mask

def get_image_with_random_shapes(width=x_dim, height=y_dim, circle_radius_max = 70):
    image = np.zeros(img_shape)
    mask = np.zeros(img_shape)

    center = (np.random.randint(0, width), np.random.randint(0, height))
    radius = np.random.randint(circle_radius_max // 2, circle_radius_max)
    color = 1  # White color for the shape
    thickness = -1  # Fill the circle
    cv2.circle(image, center, radius, color, thickness)

    # the mask should have the circle as well
    cv2.circle(mask, center, radius, color, thickness)

    top_left = (np.random.randint(0, width // 2), np.random.randint(0, height // 2))
    bottom_right = (top_left[0] + 80, top_left[1] + 120)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    '''
    # Randomly choose the number of shapes to draw
    num_shapes = np.random.randint(2, 6)
    
    for _ in range(num_shapes):
        shape_type = np.random.choice(['circle', 'rectangle', 'line'])
        
        color = 1
        thickness = -1

        # yes we are also drawing circles that are not on the mask, but they're bigger
        if shape_type == 'circle':
            # Draw a random circle
            center = (np.random.randint(0, width), np.random.randint(0, height))
            radius = np.random.randint(6, 10)
            cv2.circle(image, center, radius, color, thickness)

        elif shape_type == 'rectangle':
            # Draw a random rectangle
            top_left = (np.random.randint(0, width // 2), np.random.randint(0, height // 2))
            bottom_right = (np.random.randint(width // 2, width), np.random.randint(height // 2, height))
            cv2.rectangle(image, top_left, bottom_right, color, thickness)
        
        elif shape_type == 'line':
            # Draw a random line
            start_point = (np.random.randint(0, width), np.random.randint(0, height))
            end_point = (np.random.randint(0, width), np.random.randint(0, height))
            cv2.line(image, start_point, end_point, color, thickness=np.random.randint(1,3))
        '''
    
    return image, mask


def get_image_with_random_shape_small_mask(width=x_dim, height=y_dim):
    return get_image_with_random_shapes(width, height, circle_radius_max=5)


def get_batch(gen_img_fct, batch_size: int):
    images = []
    masks = []
    for _ in range(batch_size):
        image, mask = gen_img_fct()
        images.append(image[None, :, :])
        masks.append(mask[None, :, :])

    return np.array(images), np.array(masks)