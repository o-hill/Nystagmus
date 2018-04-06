'''

    Builds a bunch of fake images to feed to the neural
    network in order to verify the efficacy of the network.

'''

import random
import time
import numpy as np


random.seed(time.time())


def distance(x_c, y_c, x_p, y_p):
    '''

        Return the distance between the center point
        and the other point.

    '''

    x = (x_p - x_c) ** 2
    y = (y_p - y_c) ** 2

    return np.sqrt(x + y)



def create_circle():
    '''

        Create a boolean array representing a circle.

    '''

    # Get the radius of the circle and find the bounds
    # to keep the circle inside of the image.
    true_radius = random.randint(5, 20)
    upper_bound = 64 - true_radius
    lower_bound = true_radius

    # Get the real (x, y) and radius.
    true_x = random.randint(lower_bound, upper_bound)
    true_y = random.randint(lower_bound, upper_bound)

    # Create the initial image.
    image = np.zeros((64, 64), dtype=bool)

    x_list = np.arange(64)
    y_list = np.arange(64)
    X, Y = np.meshgrid(x_list, y_list)

    image[(X - true_x) ** 2 + (Y - true_y) ** 2 <= true_radius ** 2] = 1

    image = np.atleast_3d(image)

    return (image, (true_x, true_y, true_radius))





def generate_images(num_images):
    '''

        Generates <num_images> worth of boolean arrays
        that contain circles.

        Each entry in the list is a tuple:
            (image, (true_x, true_y, true_radius))

    '''

    images = []

    for i in range(num_images):
        images.append(create_circle())

    return images
