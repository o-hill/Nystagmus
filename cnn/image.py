'''

    Image class for training the network on.

    ATTENTION: Currenty using images 1 - 1141 for training from video 3,
    and holding out images 1142 - 1290 for training.

'''

from skimage.io import imread
from skimage import transform
from skimage import color
import random
import time
import numpy as np
from vessel import *
import os
import ipdb

from fake_images import generate_images

random.seed(time.time())

path = '/home/ohill/nystagmus/images/'


class Image:
    '''

        Object that reads the image data and can return a
        training and validation parition for the data.

    '''


    def __init__(self, fake):
        '''

            Grabs the images from the data directory
            and formats them for use.

        '''

        # Are we generating fake images or using real ones?
        self.fake = fake

        # Get the tuples of the actual (x, y) positions
        # for the images.
        self.vessel = Vessel('eye_data.dat')


        self.batch_size = 30
        self.images = []
        self.training = []
        self.testing = []

        new_data = []

        # Reformat the labels to match the resized images.
        for point in self.vessel.data:
            new_data.append((
                    point[0],
                    point[1] - 320,
                    point[2] - 180,
                    point[3]
            ))

        self.vessel.data = new_data

        # # If we are just testing, don't bother with the real images.
        # if self.fake:
        #    self.images = generate_images(500)

        # # Otherwise go grab them.
        # else:

        #    # Get all of the images in a list.
        #    for num in img_number:

        #        # Read all of the relevant images from our data.
        #        image = imread(path + 'nystagmus_3_{:03d}.jpg'.format(num + 1))

        #        # Resize the image so that our network fits in memory...
        #        image = np.array([i[320:960] for i in image[180:540]])
        #        image = color.rgb2gray(image)
        #        image = np.atleast_3d(image)


        #        # Add the image and its respective labels to our images array.
        #        self.images.append((image, self.vessel.data[num][1:4]))



    def get_training_partition(self):
        '''

            Splits the image list into two sub-lists,
            saving one as training and one as testing.

        '''

        # Get the numbers of the next batch to load into memory.
        training_indices = random.sample(range(0, 1140), self.batch_size)

        # Get rid of the old training examples.
        del self.training[:]

        for num in training_indices:

            image = imread(path + 'nystagmus_3_{:03d}.jpg'.format(num + 1))

            # Reformat the images to keep only the data we need.
            image = np.array([i[320:960] for i in image[180:540]])
            image = color.rgb2gray(image)
            image = np.atleast_3d(image)

            self.training.append((image, self.vessel.data[num][1:4]))

        return [i[0] for i in self.training], [i[1] for i in self.training]


    def clear_training_list(self):
        '''

            Clear the training data out of main memory.

        '''

        del self.training[:]


        # if self.fake:
        #     self.images = generate_images(500)
        #     return [i[0] for i in self.images], [i[1] for i in self.images]


#         # Randomly shuffle the image data.
#         shuff = self.images
#         random.shuffle(shuff)

#         # Use the first 80% of the list as the training data.
#         self.training = [shuff[x] for x in range(int(len(shuff) * 0.15))]

#         # Use the last 20% of the list as the holdout data.
#         self.testing = [shuff[x] for x in range(int(len(shuff) * 0.8), len(shuff))]

#         return [i[0] for i in self.training], [i[1] for i in self.training]



    def get_testing_partition(self):
        '''

            Returns the testing list.

        '''

        del self.testing[:]

        testing_indices = random.sample(range(1141, 1289), self.batch_size)

        for num in testing_indices:

            image = imread(path + 'nystagmus_3_{:03d}.jpg'.format(num + 1))
            image = np.array([i[320:960] for i in image[180:540]])
            image = color.rgb2gray(image)
            image = np.atleast_3d(image)

            self.testing.append(image)

        return self.testing, [self.vessel.data[i][1:4] for i in testing_indices], testing_indices





