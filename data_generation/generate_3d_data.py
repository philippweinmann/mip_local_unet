# %%
# in this file we want to create not 2D data, but 3d data.

import numpy as np
from matplotlib import pyplot as plt
from config import ccta_scans_dims, ccta_scans_slices

# %%
simplified_xy_dims = ccta_scans_dims // 2# for faster execution
simplified_slices = 20
x_dim = simplified_xy_dims
y_dim = simplified_xy_dims  # 256 x 256 images

img_shape = (simplified_slices, y_dim, x_dim)
print(img_shape)
# %%
ThreeDimage = np.zeros(img_shape)
# let's visualize it
# Get the indices of the '1's in the image
def visualize3Dimage(image):
    '''
    the 1s are plotted, the zeroes not.
    requires binary images
    '''
    x, y, z = np.where(image == 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the '1's in the image
    ax.scatter(x, y, z, c='red', marker='o')

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([0, image.shape[1]])
    ax.set_zlim([0, image.shape[2]])
    
    plt.show()

visualize3Dimage(ThreeDimage)
# %%
# let's start really simple. We create a tube and a cube
# first the cube
print("shape of the image: ", ThreeDimage.shape)

# let's reset the 3d image
ThreeDimage = np.zeros(img_shape)

# cube positions
left_x = np.random.randint(0, simplified_xy_dims // 2)
bottom_y = np.random.randint(0, simplified_xy_dims // 2)
right_x = left_x + 10
top_y = bottom_y + 5

print("left_x: ", left_x, " right_x: ", right_x, " bottom_y: ", bottom_y, " top_y: ", top_y)
# let's just assume the cube goes through everything
ThreeDimage[:,bottom_y:top_y, left_x:right_x] = 1
visualize3Dimage(ThreeDimage)
# %%
