import numpy as np
import math


def sigmoid_element_wise(vector_component):
    if vector_component >= 0:
        return 1 / (1 + math.exp(-vector_component))
    else:
        return math.exp(vector_component) / (math.exp(vector_component) + 1)

def sigmoid(pre_activation_vector):
    activated_vector = np.empty_like(pre_activation_vector)  # create a vector of same shape as input
    for i, elem in np.ndenumerate(pre_activation_vector):
        activated_vector[i] = sigmoid_element_wise(elem)
    return activated_vector


def relu(pre_activation_vector):
    act = np.copy(pre_activation_vector)
    act[act < 0] = 0               # get the position of vector that is -ve and make them 0
    return act


def tanh(pre_activation_vector):
    act = np.copy(pre_activation_vector)
    act = np.tanh(act)  
    return act


def activation_function(pre_activation_vector, context):
    if context == 'sigmoid':
        return sigmoid(pre_activation_vector)
    elif context == 'tanh':
        return tanh(pre_activation_vector)
    elif context == 'relu':
        return relu(pre_activation_vector)
    else:
        return None  # Error handling





