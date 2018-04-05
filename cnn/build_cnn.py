'''

    Builds a Convolutional Neural Network in Tensorflow
    and returns it from cnn().


'''


import tensorflow as tf
import numpy as np



def flatten(layer):
    '''

        Flatten the given layer and return it as a tensor.

    '''

    # Find the total number of features in the layer.
    shape = layer.get_shape()
    num_features = shape[1:4].num_elements()

    # Return the flattened layer.
    return tf.reshape(layer, [-1, num_features])




def cnn(images, Training):
    '''

        Builds and returns the CNN.

        Images:
           A tensorflow tensor of size ? x 720 x 1280 x 3

        Training:
            Boolean as to whether we are training or not.
            Used for dropout.

    '''

    # Stack a few convolutional layers on each other.
    conv1 = tf.layers.conv2d(
        inputs = images,
        filters = 32,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)


    # conv2 = tf.layers.conv2d(
    #     inputs = conv1,
    #     filters = 16,
    #     kernel_size = [5, 5],
    #     padding = 'same',
    #     activation = tf.nn.relu)


    # conv3 = tf.layers.conv2d(
    #     inputs = conv2,
    #     filters = 8,
    #     kernel_size = [5, 5],
    #     padding = 'same',
    #     activation = tf.nn.relu)


    # Pool the data with max pooling.
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [3, 3], strides = 3)


    # # Convolute and pool the data again.
    # conv4 = tf.layers.conv2d(
    #     inputs = pool1,
    #     filters = 8,
    #     kernel_size = [5, 5],
    #     padding = 'same',
    #     activation = tf.nn.relu)

    # pool2 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [3, 3], strides = 3)

    # Flatten the data and stack a couple fully connected layers for output.
    flattened = flatten(pool1)

    dense1 = tf.layers.dense(
        inputs = flattened,
        units = 32,
        activation = tf.nn.relu)

    # Dropout! Only drop features if we are training.
    drop = tf.layers.dropout(
        inputs = dense1,
        rate = 0.2,
        training = Training)

    # # Throw another dense layer and dropout in.
    # dense2 = tf.layers.dense(
    #     inputs = drop,
    #     units = 16,
    #     activation = tf.nn.relu)

    # drop2 = tf.layers.dropout(
    #     inputs = dense2,
    #     rate = 0.2,
    #     training = Training)

    # Output layer! We should be outputting (x, y).
    output = tf.layers.dense(inputs = drop, units = 3)

    return output

















