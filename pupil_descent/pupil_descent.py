"""

    A program to iteratively identify the center of an
    eye in a picture.  Using methods similar to gradient
    descent, we should be able to identify the center
    very quickly.

    Written by Oliver Hill for Michigan Aerospace.
    <oliverhi@umich.edu>

"""

import numpy as np
from vessel import Vessel
from optparse import OptionParser
import random
import time
import pylab as plt
import ipdb

random.seed(time.time())


def distance(x, y, x_c, y_c):
    return np.sqrt(((x - x_c) ** 2) + ((y - y_c) ** 2))



def find_center(x_i, y_i):

    print(x_i.shape)
    print(y_i.shape)

    # Add a dimension to the A matrix. Stack takes a tuple
    # of arrays, so there does need to be an extra pair of parentheses.
    matrix_a = np.stack((x_i, y_i, np.ones_like(x_i)))

    # Find b
    matrix_b = (-x_i ** 2) - (y_i ** 2)

    matrix_a = matrix_a.T

    matrix_x = np.linalg.lstsq(matrix_a, matrix_b)

    x_c = matrix_x[0][0] / -2
    y_c = matrix_x[0][1] / -2
    radius = np.sqrt((x_c ** 2) + (y_c ** 2) - matrix_x[0][2])

    return (x_c, y_c, radius)



def run_single_image(x_i, y_i):

    # Find the proposed center and radius of the image.
    center = find_center(x_i, y_i)

    print(center)

    distances = []

    for x, y, i in zip(x_i, y_i, range(len(x_i))):
        distances.append((distance(x, y, center[0], center[1]), i))

    plt.hist(distances, 100)
    plt.show()

    return (center, distances)





def generate_circle(n):
    '''

        Function to create a circle for the algorithm to
        test on.  It can create just the bare circle, or 
        the circle with a bunch of random points around it
        for a more realistic scenario.

    '''

    radius = random.randint(10, 300)

    theta = 2 * np.pi * np.random.rand(n)

    center = [random.randint(1, 100), random.randint(1, 100)]

    print('Real circle: (' + str(center[0]) + ', ' + str(center[1]) + ', ' + str(radius) + ')')

    r_points = [[random.randint(10, 50), random.randint(10, 50)] for i in range(200)]

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    for point in r_points:
        x = np.append(x, point[0])
        y = np.append(y, point[1])

    return x, y


def descent(image):
    '''

        Iteratively computes the center of the points in the image
        until we have a plausible solution.

        Steps:
            Find the proposed center
            Loop until the change in the center is under the threshold

    '''

    # Find the coordinate pairs in the image.
    x, y = np.where(image == True)

    # Setup and find the initial center.
    done = False
    epsilon = 0.1
    center, distances = run_single_image(x, y)

    while not done:

        # Find the average distance to the proposed center.
        raw_distances = [d[0] for d in distances]
        mean_distance = np.mean(raw_distances)
        d_sum = np.sum(raw_distances)

        norm_distances = [d/d_sum for d in raw_distances]

        # If the distance is larger than the most
        # recently computed radius, throw that point 
        # out of the calculation for the next iteration.
        for dis in distances:
            # if dis[0] > center[2] + np.std(raw_distances) or dis[0] < center[2] - np.std(raw_distances):
            if dis[0] > mean_distance + np.std(raw_distances) or dis[0] < mean_distance - np.std(raw_distances):
                x = np.delete(x, [dis[1]])
                y = np.delete(y, [dis[1]])

        # Find the new center.
        x_prev, y_prev, r_prev = center
        center, distances = run_single_image(x, y)
        change = distance(center[0], center[1], x_prev, y_prev) + (abs(center[2] - r_prev))

        # If we haven't moved, hopefully we've found the center!
        if change < epsilon or len(x) < 200:
            done = True

        print('Change: ' + str(change))

    return (center[0], center[1], center[2])







if __name__ == '__main__':

    #parser = OptionParser()
    #parser.add_option('-f', '--filename', dest='filename', help='File to read data from.')
    #(options, args) = parser.parse_args()

    # x, y = generate_circle(200)
    # print(find_center(x, y))

    # center = descent(x, y)
    # print('Done! Center: ' + str(center))

    v = Vessel('edges.vsl')
    image = random.choice(v.edges)
    center = descent(image)
    plt.imshow(image)
    plt.plot([center[0]], [center[1]], marker='o', color='red')
    plt.show()
    print(center)

   

    # run_single_image(random.choice(v.edges))

'''
&= -(p(Y = Yes|X = Fast)log_{2}p(Y = Yes|X = Fast) + p(Y = No|X = Fast)log_{2}p(Y = No|X = Fast)) \begin{comment} + 
						p(Y = Yes|X = Slow)log_{2}p(Y = Yes|X = Slow) + p(Y = No|X = Slow)log_{2}p(Y = No|X = Slow)) \end{comment}\\
'''
