'''

    Image class for training the network on.

'''

from skimage.io import imread
from skimage import transform
from skimage import color
import random
import time
import numpy as np
from vessel import *
import os
import pylab as plt

from fake_images import generate_images

random.seed(time.time())

path = '/Users/oliver/Documents/Career/Michigan_Aerospace/Eyescan/cnn_eyescan/images/'
img_number = np.arange(0, 645)


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

        self.images = []
        self.training = []
        self.testing = []

        # If we are just testing, don't bother with the real images.
        if self.fake:
           self.images = generate_images(500)

        # Otherwise go grab them.
        else:

           # Get all of the images in a list.
           for num in img_number:

               # Read all of the relevant images from our data.
               image = imread(path + 'nystagmus_3_{:03d}.jpg'.format(num + 1))

               # Resize the image so that our network fits in memory...
               image = np.array([i[320:960] for i in image[180:540]])
               image = color.rgb2gray(image)
               #image = np.atleast_3d(image)

               print(image.shape)

               plt.imshow(image, cmap='gray')
               plt.plot(self.vessel.data[num][1], self.vessel.data[num][2], marker='o', color='red')

               new_x = self.vessel.data[num][1] - 320
               new_y = self.vessel.data[num][2] - 180

               plt.plot(new_x, new_y, marker='o', color='blue')
               plt.show()


               # Add the image and its respective labels to our images array.
               self.images.append((image, self.vessel.data[num][1:4]))



    def get_training_partition(self):
        '''

            Splits the image list into two sub-lists,
            saving one as training and one as testing.

        '''

        if self.fake:
            self.images = generate_images(500)
            return [i[0] for i in self.images], [i[1] for i in self.images]


        # Randomly shuffle the image data.
        shuff = self.images
        random.shuffle(shuff)

        # Use the first 80% of the list as the training data.
        self.training = [shuff[x] for x in range(int(len(shuff) * 0.15))]

        # Use the last 20% of the list as the holdout data.
        self.testing = [shuff[x] for x in range(int(len(shuff) * 0.8), len(shuff))]

        return [i[0] for i in self.training], [i[1] for i in self.training]



    def get_testing_partition(self):
        '''

            Returns the testing list.

        '''

        return [i[0] for i in self.testing], [i[1] for i in self.testing]
