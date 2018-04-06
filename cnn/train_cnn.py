'''

    A script to train a CNN that takes a batch of images
    of eyes and identifying the center of the iris in
    each image.

'''

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import os

# Custom function to build our network.
from build_cnn import cnn

# Handles the reading and partitioning of our image data.
from image import Image

# Tests the network.
from test_cnn import test_cnn 


def restore(sess, checkpoint_path):
    '''

        Attempt to restore a network from a
        checkpoint if possible.

    '''

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    # Attempt to restore parameters for the network.
    if checkpoint:
        path = checkpoint.model_checkpoint_path
        print('Restoring model parameters from ' + str(path) + '...')

        saver.restore(sess, path)

    else:
        print('No saved model parameters found...')

    # Return checkpoint path for call to saver.save()
    save_path = os.path.join(checkpoint_path, os.path.basename(os.path.dirname(checkpoint_path)))

    return saver, save_path




def placeholders(fake = False):
    '''

        Returns the inputs as a tensor.

    '''

    if not fake:
        # A batch of images.
        images = tf.placeholder(tf.float32, [None, 360, 640, 1])

        # Corresponding labels.
        labels = tf.placeholder(tf.float32, [None, 3])

        # Indicate whether or not we are training.
        training = tf.placeholder(tf.bool)

        return images, labels, training

    else:
        images = tf.placeholder(tf.float32, [None, 64, 64, 1])
        labels = tf.placeholder(tf.float32, [None, 3])
        training = tf.placeholder(tf.bool)
        return images, labels, training


def loss_function(labels, output):
    '''

       Use mean squared error to minimize our loss.

    '''

    return tf.losses.mean_squared_error(labels, output)


def optimizer(loss):
    '''

       Use the RMS Prop optimizer to train our network.

    '''

    return tf.train.AdamOptimizer(learning_rate = 10 ** -3).minimize(loss)


def accuracy(labels, output):
    '''

        Returns the accuracy of the network as a float.

    '''

    epsilon = tf.constant(1, dtype=tf.float32)

    # The divisor for the accuracy is the number of elements in each
    # label (3) times the number of samples we are testing on.
    divisor = tf.constant(30 * 3, dtype=tf.float32)

    # Determine whether the points lie within an epsilon of our labels.
    distance = tf.abs(tf.subtract(labels, output))
    within_range = tf.less_equal(distance, epsilon)

    # Return the reduced sum over our divisor.
    return tf.divide(tf.reduce_sum(tf.cast(within_range, dtype=tf.float32)), divisor)


def get_tensor_batch_size(tensor):
    '''

        Returns the batch size of the given tensor.

    '''

    return tensor.get_shape()[0].value


def run():
    '''

        Build and train our network.

    '''

    # Get the pipeline and build the graph.
    place_images, place_labels, training = placeholders(fake=False)
    output = cnn(place_images, training)

    # Get the loss and optimizer.
    loss = loss_function(place_labels, output)
    train_op = optimizer(loss)

    acc = accuracy(place_labels, output)

    # Run the network.
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())

       saver, save_path = restore(sess, './checkpoints/')

       # Read the image data.
       data = Image(fake=False)

       # Run for 1000 rounds?
       training_round = np.arange(0, 1000000)
       for batch in training_round:

           print('Batch ' + str(batch))

           # Get the training and testing labels.
           train_images, train_labels = data.get_training_partition()
           test_images, test_labels, test_indices = data.get_testing_partition()

           print('Images/labels found')

           # Create the feed dictionary of the placeholders to the data.
           fd = {
                place_images: train_images,
                place_labels: train_labels,
                training: True
            }

           print('Running...')

           # Run the session.
           _, sess_loss = sess.run([train_op, loss], feed_dict = fd)
           print('> Training loss at iteration ' + str(batch) + ': ' + str(sess_loss))

           # Save the parameters.
           if batch % 50 == 0:
               print('Saving...')
               saver.save(sess, save_path)

           if batch % 1000 == 0:
               print('Testing...')
               data.clear_training_list()
               test_cnn(sess, training, place_images, place_labels, acc, loss, output, \
                       test_images, test_labels, test_indices, batch)



if __name__ == '__main__':
    run()

