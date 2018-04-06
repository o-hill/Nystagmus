'''

    Functions for testing the Nystagmus network.

'''


import tensorflow as tf

# Use matplotlib without a display.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os



def test_cnn(sess, training, images, labels, acc, loss, output, test_images, test_labels, test_indices, batch):

    # Use the testing data for our validation.
    fd = {
        images: test_images,
        labels: test_labels,
        training: False    
    }

    # Run the session.
    test_pred, test_acc, test_loss = sess.run([output, acc, loss], feed_dict=fd)
    
    # Log our progress in the terminal.
    log_testing_progress(batch, test_acc, test_loss)

    # Create images with the predicted centers to see how we are doing.
    create_testing_feedback(test_pred, test_labels, test_images, test_indices, batch)




def log_testing_progress(batch, test_acc, test_loss):
    '''

        Print the testing progress to the terminal.

    '''

    print('>> Testing progress at iteration ' + str(batch))
    print('\tAccuracy: ' + str(test_acc))
    print('\tLoss: ' + str(test_loss))



def create_testing_feedback(test_pred, test_labels, test_images, test_indices, batch):
    '''

        Creates an image for each testing prediction with the 
        real label plotted over it as a blue dot and the predicted
        label as a red dot.  The images go in a sub-directory called
        'predictions'.

    '''

    current_dir = os.getcwd()
    save_dir = current_dir + '/predictions/batch_' + str(int(batch / 1000))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0
    for image, num in zip(test_images, test_indices):

        im_name = save_dir + '/pred_' + str(num + 1) + '.jpg'

        # Plot the image and the two labels.
        plt.imshow(image.squeeze(), cmap='gray')
        plt.plot([test_pred[i][0]], [test_pred[i][1]], marker='o', color='red')
        plt.plot([test_labels[i][0]], [test_labels[i][1]], marker='o', color='blue')

        # Finally, let's save the image!
        plt.savefig(im_name)

        # Clear out the figure for the next image.
        plt.close('all')

        i += 1

    print('Done creating testing images!')

        










