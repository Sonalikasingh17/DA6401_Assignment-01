import numpy as np
import math
import wandb
from optimiser import *
from tensorflow.keras.datasets import fashion_mnist


""" get training and testing vectors
    Number of Training Images = 60000
    Number of Testing Images = 10000 """
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


# # Convert labels to one-hot encoding
# def one_hot_encode(labels, num_classes):
#     return np.eye(num_classes)[labels]

# y_train = one_hot_encode(y_train, 10)
# y_test = one_hot_encode(y_test, 10)

# from loss import *
# from gradient import *
# from activation import *
# import copy
# from keras.preprocessing.image import ImageDataGenerator


# # Define the neural network class

