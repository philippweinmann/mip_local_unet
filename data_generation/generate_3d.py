# %%
# in this file we want to create not 2D data, but 3d data.

import numpy as np
from matplotlib import pyplot as plt
from data_generation.config import original_image_shape

# %%
class Image3d():
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.image = np.zeros(img_shape)
        self.mask = np.zeros(img_shape)
        self.generate_network_input()

    def visualize3Dimage(self):
        '''
        the 1s are plotted, the zeroes not.
        requires binary images
        '''
        x, y, z = np.where(self.image == 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the '1's in the image
        ax.scatter(x, y, z, c='red', marker='o')

        # Set labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([0, self.image.shape[0]])
        ax.set_ylim([0, self.image.shape[1]])
        ax.set_zlim([0, self.image.shape[2]])
        
        plt.show()
    
    # cube positions
    def add_cube(self):
        z_dim = self.image.shape[0]
        y_dim = self.image.shape[1]
        x_dim = self.image.shape[2]

        left_x = np.random.randint(0, x_dim // 2)
        bottom_y = np.random.randint(0, y_dim // 2)
        right_x = left_x + 10
        top_y = bottom_y + 5

        # print("left_x: ", left_x, " right_x: ", right_x, " bottom_y: ", bottom_y, " top_y: ", top_y)
        # let's just assume the cube goes through everything
        self.image[:,bottom_y:top_y, left_x:right_x] = 1

    def add_tube(self):
        z_dim = self.image.shape[0]
        y_dim = self.image.shape[1]
        x_dim = self.image.shape[2]

        # circle_center = (y_center, x_center)
        x_center = np.random.randint(0, x_dim)
        y_center = np.random.randint(0, y_dim)
        
        radius = 20

        # let's create a grid of x and y coordinates
        y, x = np.ogrid[0:y_dim, 0:x_dim]
        distance_from_center = (y - y_center) ** 2 + (x - x_center) ** 2
        circle_mask = distance_from_center <= radius * 2

        self.image[:,circle_mask] = 1
        self.mask[:,circle_mask] = 1

    def generate_network_input(self):
        self.add_cube()
        self.add_tube()

    
simplified_xy_dims = original_image_shape[1] // 16 # for faster execution
simplified_slices = simplified_xy_dims # the Unet requires an input that can be divided by 16???
x_dim = simplified_xy_dims
y_dim = simplified_xy_dims  # 256 x 256 images

img_shape = (simplified_slices, y_dim, x_dim)
# %%

def get_3DImage(img_shape = img_shape): 
    image3d = Image3d(img_shape)

    return image3d.image, image3d.mask

def get_batch(gen_img_fct, img_shape = img_shape, batch_size: int = 10):
    images = []
    masks = []
    for _ in range(batch_size):
        image, mask = gen_img_fct(img_shape = img_shape)
        images.append(image[None, :, :])
        masks.append(mask[None, :, :])

    return np.array(images), np.array(masks)