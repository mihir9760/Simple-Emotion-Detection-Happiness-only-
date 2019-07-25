import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def mean_predctn(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    dataset = h5py.File('datasets/train.h5', "r")
    train_set_x = np.array(dataset["train_set_x"][:]) # your train set features
    train_set_y = np.array(dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test.h5', "r")
    test_set_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    
    return train_set_x, train_set_y, test_set_x, test_set_y, classes

