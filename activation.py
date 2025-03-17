import numpy as np


# Activation functions and its derivatives

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    return  (1.0 / (1 + np.exp(-(x))))*(1 -  1.0 / (1 + np.exp(-(x))))

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x) 
def relu_derivative(x):
    return (x>0)*1 

