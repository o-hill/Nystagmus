'''

    A script to train a CNN that takes a batch of images
    of eyes and identifying the center of the iris in
    each image.

'''


import tensorflow as tf
import numpy as np
import os

# Custom function to build our network.
from build_cnn import cnn

# Handles the reading and partitioning of our image data.
from image import Image


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




def placeholders():
    '''

        Returns the inputs as a tensor.

    '''

    # A batch of images.
    images = tf.placeholder(tf.float32, [None, 720, 1280, 3])

    # Corresponding labels.
    labels = tf.placeholder(tf.float32, [None, 3])

    return images, labels


def loss_function(labels, output):
    '''

       Use mean squared error to minimize our loss.

    '''

    return tf.losses.mean_squared_error(labels, output)


def optimizer(loss):
    '''

       Use the RMS Prop optimizer to train our network.

    '''

    return tf.train.RMSPropOptimizer(learning_rate = 10 ** -3).minimize(loss)



def run():
    '''

        Build and train our network.

    '''

    # Get the pipeline and build the graph.
    place_images, place_labels = placeholders()
    output = cnn(place_images, Training = True)

    # Get the loss and optimizer.
    loss = loss_function(place_labels, output)
    train_op = optimizer(loss)

    # Run the network.
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())

       saver, save_path = restore(sess, './checkpoints/')

       # Read the image data.
       data = Image()

       # Run for 1000 rounds?
       training_round = np.arange(0, 1000)
       for batch in training_round:

           print('Batch ' + str(batch))

           # Get the training and testing labels.
           train_images, train_labels = data.get_training_partition()
           test_images, test_labels = data.get_testing_partition()

           print('Images/labels found')

           # Create the feed dictionary of the placeholders to the data.
           fd = { place_images: train_images, place_labels: train_labels }

           print('Running...')

           # Run the session.
           _, sess_loss = sess.run([train_op, loss], feed_dict = fd)
           print('> Training loss at iteration ' + str(batch) + ': ' + str(sess_loss))

           # Save the parameters.
           if batch % 50 == 0:
               print('Saving...')
               saver.save(sess, save_path)



if __name__ == '__main__':
    run()

