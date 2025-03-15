import numpy as np


# Activation Funtions and its derivatives


def sigmoid(z):
    return 1.0 / (1 + np.exp(-(z)))
def sigmoid_derivative(z):
    return  (1.0 / (1 + np.exp(-(z))))*(1 -  1.0 / (1 + np.exp(-(z))))

def tanh(z):
    return np.tanh(z)
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0.001, z) 
def relu_derivative(z):
    return (z>0)*1 + (z<0)*0.001 



