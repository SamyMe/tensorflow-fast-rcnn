# -*- coding: utf-8 -*-
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_shapes(img, faces=[], lp=[], shape=None, img_name="img.jpg"):
    """
    img : ndimage
    shape : "rectangle", "circle", "polygn"
    coordinats : rectangle => (x, y, dx, dy)
                 circle => (x, y, dx, dy)
    """

    # If changed dimensions, re put them
    # print img.shape
    if img.shape[2]>3:
        img = np.rollaxis(img, 0, 3)
    # print img.shape

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Converting coordinates 
    for shape in [faces, lp]:
        for coordinate in shape:
            x = coordinate[0]
            y = coordinate[1]
            dx = coordinate[2]
            dy = coordinate[3]

            shape = patches.Rectangle((x,y), width=dx, height=dy, linewidth=2,
                                        edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(shape)

    # plt.show()
    plt.savefig(img_name, dpi=900)
