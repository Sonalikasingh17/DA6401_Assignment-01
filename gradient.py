import numpy as np

'''In this file I calculate gradient for the loss function and activation function'''


def cross_entropy_grad(y_hat, label):
    loss_grad = np.zeros_like(y_hat)
    if y_hat[label] < 10 ** -8:
        y_hat[label] = 10 ** -8
    loss_grad[label] = -1 / (y_hat[label])
    norm = np.linalg.norm(loss_grad)
    if norm > 100.0:
        return loss_grad * 100.0 / norm
    else:
        return loss_grad


def squared_error_grad(y_hat, label):
    loss_grad = np.copy(y_hat)
    loss_grad[label] -= 1
    loss_grad = 2 * loss_grad
    loss_grad = loss_grad / len(y_hat)
    norm = np.linalg.norm(loss_grad)
    if norm > 100.0:
        return loss_grad * 100.0 / norm
    else:
        return loss_grad


def output_grad(y_hat, label, loss_type):
    if loss_type == 'cross_entropy':
        return cross_entropy_grad(y_hat=y_hat, label=label)
    elif loss_type == 'squared_error':
        return squared_error_grad(y_hat=y_hat, label=label)

# calculation of gradient w.r.t last layer
def last_grad(y_hat, label):                   
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


# calculation of gradient w.r.t 'a_i's (activation function sigmoid) after h_i
def sigmoid_grad(post_activation):
    return np.multiply(post_activation, 1 - post_activation)


# calculation of gradient w.r.t 'a_i's (activation function tanh) after h_i
def tanh_grad(post_activation):
    return 1 - np.power(post_activation, 2)


# calculation of gradient w.r.t 'a_i''s (activation function relu) after h_i
def relu_grad(pre_activation_vector):
    grad = np.copy(pre_activation_vector)
    # making +ve and 0 component 1
    grad[grad >= 0] = 1
    # making -ve component 0
    grad[grad < 0] = 0
    return grad


def a_grad(network, delta, layer):
    # Gradient w.r.t  a_i's (pre_activation)
    if network[layer]['context'] == 'sigmoid':
        active_grad_ = sigmoid_grad(network[layer]['h'])
    elif network[layer]['context'] == 'tanh':
        active_grad_ = tanh_grad(network[layer]['h'])
    elif network[layer]['context'] == 'relu':
        active_grad_ = relu_grad(network[layer]['a'])

    grad_value = np.multiply(delta[layer]['h'], active_grad_)
    norm = np.linalg.norm(grad_value)
    if norm > 100.0:
        return grad_value * 100.0 / norm
    else:
        return grad_value
    


def h_grad(network, delta, layer):
    # Gradient w.r.t  h_i (activation)
    network[layer]['weight'].transpose()
    grad_value = network[layer + 1]['weight'].transpose() @ delta[layer + 1]['a']
    norm = np.linalg.norm(grad_value)
    if norm > 100.0:
        return grad_value * 100.0 / norm
    else:
        return grad_value


def w_grad(network, delta, layer, x):
    # Gradient w.r.t w_i's (weight matrix)
    if layer == 0:
        grad_value = delta[layer]['a'] @ x.transpose()
    else:
        grad_value = delta[layer]['a'] @ network[layer - 1]['h'].transpose()
    norm = np.linalg.norm(grad_value)
    if norm > 10000.0:
        return grad_value * 10000.0 / norm
    else:
        return grad_value
    

def b_grad(delta, layer):
    # Gradient w.r.t b_i's (bias vector)
    return delta[layer]['a']