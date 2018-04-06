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


def dropout(layer, training, drop_rate):
    '''

        Decide whether or not to use dropout based on whether or not
        we are training.

    '''

    return tf.cond(training, lambda: drop(layer, drop_rate), lambda: no_drop(layer))


def drop(layer, drop_rate):
    '''
    
        Return the dropout layer.

    '''

    return tf.layers.dropout(layer, rate=drop_rate)


def no_drop(layer):
    '''

        We are not training, so return just the layer.

    '''

    return layer


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
        inputs = dropout(images, Training, 0.05),
        filters = 64,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)


    conv2 = tf.layers.conv2d(
        inputs = dropout(conv1, Training, 0.10),
        filters = 32,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)

    # Pool the data with max pooling.
    pool1 = tf.layers.max_pooling2d(inputs = dropout(conv2, Training, 0.15),
            pool_size = [3, 3], strides = 3)

    conv3 = tf.layers.conv2d(
        inputs = dropout(pool1, Training, 0.20),
        filters = 16,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs = dropout(conv3, Training, 0.25),
        filters = 8,
        kernel_size = [5, 5],
        padding = 'same',
        activation = tf.nn.relu)

    # Pool the data with max pooling.
    pool2 = tf.layers.max_pooling2d(inputs = dropout(conv2, Training, 0.30),
            pool_size = [3, 3], strides = 3)

    # Flatten the data and stack a couple fully connected layers for output.
    flattened = flatten(pool1)

    dense1 = tf.layers.dense(
        inputs = flattened,
        units = 128,
        activation = tf.nn.relu)

    dense2 = tf.layers.dense(
        inputs = flattened,
        units = 64,
        activation = tf.nn.relu)

    # Output layer! We should be outputting (x, y, radius).
    output = tf.layers.dense(inputs = dense2, units = 3)

    return output



'''
    # Throw another dense layer and dropout in.
    dense2 = tf.layers.dense(
	inputs = drop,
	units = 16,
	activation = tf.nn.relu)

    drop2 = tf.layers.dropout(
	inputs = dense2,
	rate = 0.2,
	training = Training)
'''














