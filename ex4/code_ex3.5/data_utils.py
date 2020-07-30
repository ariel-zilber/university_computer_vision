import tensorflow as tf
import numpy as np


#TODO:
def load_MNIST():
    '''
    Loads the MNIST dataset of 0/1 digits
    :return:
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_filter = np.where((y_train == 0 ) | (y_train == 1))
    test_filter = np.where((y_test == 0) | (y_test == 1))
    x_train, y_train = x_train[train_filter], y_train[train_filter]
    x_test, y_test = x_test[test_filter], y_test[test_filter]

    return (x_train, y_train), (x_test, y_test)

def normalize_dataset(data_set,rows,cols,channels):
    '''
    Normaalizes a given data set
    :param data_set:
    :param rows:
    :param cols:
    :param channels:
    :return:
    '''
    data_set = data_set.reshape(data_set.shape[0], rows, cols, channels)
    data_set = data_set.astype('float32')
    data_set/=255
    return data_set