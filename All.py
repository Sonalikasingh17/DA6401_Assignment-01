

# grad.py
import numpy as np
def cross_entropy_grad(y_hat, label):
    # grad w.r.t out activation
    temp = np.zeros_like(y_hat)
    # If the initial guess is very wrong. This gradient will explode. This places a limit on that.
    if y_hat[label] < 10 ** -8:
        y_hat[label] = 10 ** -8
    temp[label] = -1 / (y_hat[label])
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp

def squared_error_grad(y_hat, label):
    # grad w.r.t out activation
    temp = np.copy(y_hat)
    temp[label] -= 1
    temp = 2 * temp
    temp = temp / len(y_hat)
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


def output_grad(y_hat, label, loss_type):
    if loss_type == 'cross_entropy':
        return cross_entropy_grad(y_hat=y_hat, label=label)
    elif loss_type == 'squared_error':
        return squared_error_grad(y_hat=y_hat, label=label)


def last_grad(y_hat, label):
    # grad w.r.t out last layer
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp
# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is sigmoid.We have passed h_is
def sigmoid_grad(post_activation):
    return np.multiply(post_activation, 1 - post_activation)
# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is tanh. We have passed h_is
def tanh_grad(post_activation):
    return 1 - np.power(post_activation, 2)
# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is relu.
def relu_grad(pre_activation_vector):
    grad = np.copy(pre_activation_vector)
    # making +ve and 0 component 1
    grad[grad >= 0] = 1
    # making -ve component 0
    grad[grad < 0] = 0
    return grad

def a_grad(network, transient_gradient, layer):
    # grad w.r.t  a_i's layer
    if network[layer]['context'] == 'sigmoid':
        active_grad_ = sigmoid_grad(network[layer]['h'])
    elif network[layer]['context'] == 'tanh':
        active_grad_ = tanh_grad(network[layer]['h'])
    elif network[layer]['context'] == 'relu':
        active_grad_ = relu_grad(network[layer]['a'])
    temp = np.multiply(transient_gradient[layer]['h'], active_grad_)
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp
    # hadamard multiplication
def h_grad(network, transient_gradient, layer):
    # grad w.r.t out h_i layer
    network[layer]['weight'].transpose()
    temp = network[layer + 1]['weight'].transpose() @ transient_gradient[layer + 1]['a']
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp
def w_grad(network, transient_gradient, layer, x):
    if layer == 0:
        temp = transient_gradient[layer]['a'] @ x.transpose()
    else:
        temp = transient_gradient[layer]['a'] @ network[layer - 1]['h'].transpose()
    norm = np.linalg.norm(temp)
    if norm > 10000.0:
        return temp * 10000.0 / norm
    else:
        return temp
    
#activation.py
import numpy as np
import math
# this function helps in calculation of sigmoid function value of a component of vector
def sigmoid_element_wise(vector_component):
    # if-else to prevent math overflow
    if vector_component >= 0:
        return 1 / (1 + math.exp(-vector_component))
    else:
        return math.exp(vector_component) / (math.exp(vector_component) + 1)

# this function calculated sigmoid of pre - activation layer
def sigmoid(pre_activation_vector):
    # create a vector of same shape as input
    activated_vector = np.empty_like(pre_activation_vector)
    # iterate over input
    for i, elem in np.ndenumerate(pre_activation_vector):
        # calculate component wise sigmoid
        activated_vector[i] = sigmoid_element_wise(elem)
    return activated_vector
# this function calculates softmax
def softmax(pre_activation_vector):
    post_act = np.copy(pre_activation_vector)
    # we are shifting the value of exponent because in case of large error, there can be nan problem,this is the fix
    max_exponent = np.max(post_act)
    post_act = np.exp(post_act - max_exponent)
    post_act = post_act / np.sum(post_act)
    return post_act
# this function calculates softmax
def relu(pre_activation_vector):
    post_act = np.copy(pre_activation_vector)
    # get the position of vector that is -ve and make them 0
    post_act[post_act < 0] = 0
    return post_act
# this function handles the input and redirects the request to proper function
def activation_function(pre_activation_vector, context):
    if context == 'softmax':
        # calling softmax
        return softmax(pre_activation_vector)
    elif context == 'sigmoid':
        # calling sigmoid
        return sigmoid(pre_activation_vector)
    elif context == 'tanh':
        # creating tanh
        return np.copy(np.tanh(pre_activation_vector))
    elif context == 'relu':
        # calling relu
        return relu(pre_activation_vector)
    else:
        # Error handling
        return None
    
#optimiser.py
import sys
import copy
import math
import numpy as np

"""This file contains various gradient optimisers"""


# class for simple gradient descent
class SimpleGradientDescent:
    def __init__(self, eta, layers, weight_decay=0.0):
        # learning rate
        self.eta = eta
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # learning rate controller
        self.lrc = 1.0
        # weight decay
        self.weight_decay = weight_decay

    # function for gradient descending
    def descent(self, network, gradient):
        for i in range(self.layers):
            network[i]['weight'] = network[i]['weight'] - ((self.eta / self.lrc) * gradient[i][
                'weight']) - (self.eta * self.weight_decay * network[i]['weight'])
            network[i]['bias'] -= ((self.eta / self.lrc) * gradient[i]['bias'])
        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


# class for Momentum gradient descent
class MomentumGradientDescent:
    def __init__(self, eta, layers, gamma, weight_decay=0.0):
        # learning rate
        self.eta = eta
        self.gamma = gamma
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # rate learning controller
        self.lrc = 1
        # historical momentum
        self.momentum = None
        # weight decay
        self.weight_decay = weight_decay

    # function for gradient descending
    def descent(self, network, gradient):
        """http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , Slide 70"""
        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            # initialize momentum- refer above lecture slide 36
            for i in range(self.layers):
                self.momentum[i]['weight'] = (self.eta / self.lrc) * gradient[i]['weight']
                self.momentum[i]['bias'] = (self.eta / self.lrc) * gradient[i]['bias']
        else:
            # update momentum
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + (self.eta / self.lrc) * gradient[i][
                    'weight']
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + (self.eta / self.lrc) * gradient[i][
                    'bias']
        # the descent
        for i in range(self.layers):
            network[i]['weight'] = network[i]['weight'] - self.momentum[i]['weight'] - (
                        (self.eta / self.lrc) * self.weight_decay * network[i][
                    'weight'])
            network[i]['bias'] -= self.momentum[i]['bias']

        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


# class for NAG
class NAG:
    def __init__(self, eta, layers, gamma, weight_decay=0.0):
        # learning rate
        self.eta = eta
        self.gamma = gamma
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # historical momentum
        self.momentum = None
        # learning rate controller
        self.lrc = 1.0
        # weight decay
        self.weight_decay = weight_decay

    # function for lookahead. Call this before forward propagation.
    def lookahead(self, network):
        # case when no momentum has been generated yet.
        if self.momentum is None:
            pass
        else:
            # update the gradient using momentum
            for i in range(self.layers):
                network[i]['weight'] -= self.gamma * self.momentum[i]['weight']
                network[i]['bias'] -= self.gamma * self.momentum[i]['bias']

    # function for gradient descending
    def descent(self, network, gradient):

        # the descent
        for i in range(self.layers):
            network[i]['weight'] = network[i]['weight'] - ((self.eta / self.lrc) * gradient[i][
                'weight']) - ((self.eta / self.lrc) * self.weight_decay * network[i]['weight'])
            network[i]['bias'] -= self.eta * gradient[i]['bias']

        gamma = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.gamma)

        # generate momentum for the next time step next

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            # initialize momentum
            for i in range(self.layers):
                self.momentum[i]['weight'] = (self.eta / self.lrc) * gradient[i]['weight']
                self.momentum[i]['bias'] = (self.eta / self.lrc) * gradient[i]['bias']
        else:
            # update momentum: http://cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture5.pdf , slide: 46
            for i in range(self.layers):
                self.momentum[i]['weight'] = gamma * self.momentum[i]['weight'] + ((self.eta / self.lrc) * gradient[i][
                    'weight'])
                self.momentum[i]['bias'] = gamma * self.momentum[i]['bias'] + (
                            (self.eta / self.lrc) * gradient[i]['bias'])

        self.calls += 1
        if self.calls % 10 == 0:
            self.lrc += 1.0


"""As mentioned in this paper: https://arxiv.org/pdf/1609.04747.pdf 
RMSProp, ADAM and NADAM have adaptive learning rates so they do not need a lrc"""


class RMSProp:
    def __init__(self, eta, layers, beta, weight_decay=0.0):
        # learning rate
        self.eta = eta
        # decay parameter for denominator
        self.beta = beta
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # epsilon
        self.epsilon = 0.001
        # to implement update rule for RMSProp
        self.update = None
        # weight decay
        self.weight_decay = weight_decay

    # function for gradient descending
    def descent(self, network, gradient):

        # generate update for the next time step
        if self.update is None:
            # copy the structure
            self.update = copy.deepcopy(gradient)
            # initialize update at time step 1 assuming that update at time step 0 is 0
            for i in range(self.layers):
                self.update[i]['weight'] = (1 - self.beta) * (gradient[i]['weight']) ** 2
                self.update[i]['bias'] = (1 - self.beta) * (gradient[i]['bias']) ** 2
        else:
            for i in range(self.layers):
                self.update[i]['weight'] = self.beta * self.update[i]['weight'] + (1 - self.beta) * (gradient[i][
                    'weight']) ** 2
                self.update[i]['bias'] = self.beta * self.update[i]['bias'] + (1 - self.beta) * (
                    gradient[i]['bias']) ** 2
        # Now we use the update rule for RMSProp
        for i in range(self.layers):
            network[i]['weight'] = network[i]['weight'] - np.multiply(
                (self.eta / np.sqrt(self.update[i]['weight'] + self.epsilon)),
                gradient[i]['weight']) - self.weight_decay * network[i]['weight']
            network[i]['bias'] = network[i]['bias'] - np.multiply(
                (self.eta / np.sqrt(self.update[i]['bias'] + self.epsilon)), gradient[i]['bias'])

        self.calls += 1


# class for ADAM: Reference: https://arxiv.org/pdf/1412.6980.pdf?source=post_page---------------------------
"""Using the previous gradients instead of the previous updates allows the algorithm to continue changing 
   direction even when the learning rate has annealed significantly toward the end of training, resulting 
   in more precise fine-grained convergence"""


class ADAM:
    def __init__(self, eta, layers, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        # learning rate
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # first moment vector m_t: defined as a decaying mean over the previous gradients
        self.momentum = None
        self.t_momentum = None
        # second moment vector v_t
        self.second_momentum = None
        self.t_second_momentum = None
        # epsilon
        self.eps = eps
        # weight decay
        self.weight_decay = weight_decay

    # function for gradient descending
    def descent(self, network, gradient):

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            self.second_momentum = copy.deepcopy(gradient)
            for i in range(self.layers):
                # first momentum initialization
                self.momentum[i]['weight'][:] = np.zeros_like(gradient[i]['weight'])
                self.momentum[i]['bias'][:] = np.zeros_like(gradient[i]['bias'])
                # second momentum initialization
                self.second_momentum[i]['weight'][:] = np.zeros_like(gradient[i]['weight'])
                self.second_momentum[i]['bias'][:] = np.zeros_like(gradient[i]['bias'])
            self.t_momentum = copy.deepcopy(self.momentum)
            self.t_second_momentum = copy.deepcopy(self.second_momentum)

        for i in range(self.layers):
            # Update biased first moment estimate: Moving average of gradients
            self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * gradient[i][
                'weight']
            self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i]['bias'
            ]
            # Update biased second raw moment estimate: rate adjusting parameter update similar to RMSProp
            self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (
                    1 - self.beta2) * np.power(gradient[i][
                                                   'weight'], 2)
            self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (
                    1 - self.beta2) * np.power(gradient[i]['bias'
                                               ], 2)
        # bias correction
        for i in range(self.layers):
            self.t_momentum[i]['weight'][:] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['weight']
            self.t_momentum[i]['bias'][:] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias']

            self.t_second_momentum[i]['weight'][:] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i][
                'weight']
            self.t_second_momentum[i]['bias'][:] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i][
                'bias']

        # the descent
        for i in range(self.layers):
            # temporary variable for calculation
            temp = np.sqrt(self.t_second_momentum[i]['weight'])
            # add epsilon to square root of temp
            temp_eps = temp + self.eps
            # inverse everything
            temp_inv = 1 / temp_eps
            # perform descent: Update rule for weight along with l2 regularisation
            network[i]['weight'] = network[i]['weight'] - self.eta * (
                np.multiply(temp_inv, self.t_momentum[i]['weight'])) - (
                                               self.eta * self.weight_decay * network[i]['weight'])

            # now we do the same for bias
            # temporary variable for calculation
            temp = np.sqrt(self.t_second_momentum[i]['bias'])
            # add epsilon to square root of temp
            temp_eps = temp + self.eps
            # inverse everything
            temp_inv = 1 / temp_eps
            # perform descent for weight
            network[i]['bias'] -= self.eta * np.multiply(temp_inv, self.t_momentum[i]['bias'])

        self.calls += 1


# Reference: https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
class NADAM:
    def __init__(self, eta, layers, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        # learning rate
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        # number of layers
        self.layers = layers
        # number of calls
        self.calls = 1
        # first moment vector m_t: defined as a decaying mean over the previous gradients
        self.momentum = None
        # second moment vector v_t
        self.second_momentum = None
        # epsilon
        self.eps = eps
        # weight decay
        self.weight_decay = weight_decay

    # function for gradient descending: Algorithm 2 Page 3
    def descent(self, network, gradient):

        if self.momentum is None:
            # copy the structure
            self.momentum = copy.deepcopy(gradient)
            self.second_momentum = copy.deepcopy(gradient)
            # initialize momentums
            for i in range(self.layers):
                # first momentum initialization
                self.momentum[i]['weight'] = (1 - self.beta1) * gradient[i]['weight']
                self.momentum[i]['bias'] = (1 - self.beta1) * gradient[i]['bias']
                # second momentum initialization
                self.second_momentum[i]['weight'] = (1 - self.beta2) * np.power(gradient[i]['weight'], 2)
                self.second_momentum[i]['bias'] = (1 - self.beta2) * np.power(gradient[i]['bias'], 2)
        else:
            for i in range(self.layers):
                # Update biased first moment estimate: Moving average of gradients
                self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * \
                                             gradient[i][
                                                 'weight']
                self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i][
                    'bias'
                ]
                # Update biased second raw moment estimate: rate adjusting parameter update similar to RMSProp
                self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (
                        1 - self.beta2) * np.power(gradient[i][
                                                       'weight'], 2)
                self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (
                        1 - self.beta2) * np.power(gradient[i]['bias'
                                                   ], 2)
        # bias correction
        m_t_hat = copy.deepcopy(self.momentum)
        v_t_hat = copy.deepcopy(self.second_momentum)
        for i in range(self.layers):
            m_t_hat[i]['weight'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i][
                'weight'] + ((1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['weight']
            m_t_hat[i]['bias'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias'] + (
                    (1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['bias']

            v_t_hat[i]['weight'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * \
                                   self.second_momentum[i][
                                       'weight']
            v_t_hat[i]['bias'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i][
                'bias']

        # the descent
        for i in range(self.layers):
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['weight'] + self.eps)
            # inverse everything
            temp_inv = 1 / temp
            # perform descent for weight
            network[i]['weight'] = network[i]['weight'] - self.eta * (
                np.multiply(temp_inv, m_t_hat[i]['weight'])) - (self.eta * self.weight_decay * network[i]['weight'])

            # now we do the same for bias
            # temporary variable for calculation
            temp = np.sqrt(self.second_momentum[i]['bias']) + self.eps
            # inverse everything
            temp_inv = 1 / temp
            # perform descent for weight
            network[i]['bias'] -= self.eta * np.multiply(temp_inv, v_t_hat[i]['bias'])

        self.calls += 1

#loss.py
"""This file will contain various methods for calculation of loss functions"""
import numpy as np

# calculate cross entropy
def cross_entropy(label, softmax_output):
    # as we have only one true label, we have simplified the function for faster calculation.
    if softmax_output[label] < 10 ** -8:
        return -np.log(10 ** -8)
    return -np.log(softmax_output[label])


def squared_error(label, softmax_output):
    true_vector = np.zeros_like(softmax_output)
    true_vector[label] = 1
    size = float(len(softmax_output))
    return np.array([(np.linalg.norm(true_vector - softmax_output) ** 2) / size]).reshape((1, 1))

#main.py
"""Implement Feed Forward neural network where the parameters are
   number of hidden layers and number of neurons in each hidden layer"""
from loss import *
from grad import *
from activation import *
from optimiser import *
import copy
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import wandb

""" get training and testing vectors
    Number of Training Images = 60000
    Number of Testing Images = 10000 """
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

last = 2
# network is a list of all the learning parameters in every layer and gradient is its copy
network = []
gradient = []
# store gradient w.r.t a single datapoint
transient_gradient = []
# will contain the total amount of loss for each timestep(1). One timestep is defined as one update of the parameters.
loss = 0


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        network[i]['h'] = activation_function(network[i]['a'], context=network[i]['context'])


def backward_propagation(number_of_layers, x, y, number_of_datapoint, loss_type, clean=False):
    transient_gradient[number_of_layers - 1]['h'] = output_grad(network[number_of_layers - 1]['h'], y,
                                                                loss_type=loss_type)
    transient_gradient[number_of_layers - 1]['a'] = last_grad(network[number_of_layers - 1]['h'], y)
    for i in range(number_of_layers - 2, -1, -1):
        transient_gradient[i]['h'] = h_grad(network=network, transient_gradient=transient_gradient, layer=i)
        transient_gradient[i]['a'] = a_grad(network=network, transient_gradient=transient_gradient, layer=i)
    for i in range(number_of_layers - 1, -1, -1):
        transient_gradient[i]['weight'] = w_grad(network=network, transient_gradient=transient_gradient, layer=i, x=x)
        transient_gradient[i]['bias'] = gradient[i]['a']
    if clean:
        gradient[number_of_layers - 1]['h'] = transient_gradient[number_of_layers - 1]['h'] / float(number_of_datapoint)
        gradient[number_of_layers - 1]['a'] = transient_gradient[number_of_layers - 1]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            gradient[i]['h'] = transient_gradient[i]['h'] / float(number_of_datapoint)
            gradient[i]['a'] = transient_gradient[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            gradient[i]['weight'] = transient_gradient[i]['weight'] / float(number_of_datapoint)
            gradient[i]['bias'] = transient_gradient[i]['bias'] / float(number_of_datapoint)
    else:

        gradient[number_of_layers - 1]['h'] += transient_gradient[number_of_layers - 1]['h'] / float(
            number_of_datapoint)
        gradient[number_of_layers - 1]['a'] += transient_gradient[number_of_layers - 1]['a'] / float(
            number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            gradient[i]['h'] += transient_gradient[i]['h'] / float(number_of_datapoint)
            gradient[i]['a'] += transient_gradient[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            gradient[i]['weight'] += transient_gradient[i]['weight'] / float(number_of_datapoint)
            gradient[i]['bias'] += transient_gradient[i]['bias'] / float(number_of_datapoint)


# this function is used for validation, useful during hyperparameter tuning or model change.
def validate(number_of_layer, validateX, validateY, loss_type):
    loss_local = 0
    acc = 0
    if loss_type == 'cross_entropy':
        for x, y in zip(validateX, validateY):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding loss w.r.t to a single datapoint
            loss_local += cross_entropy(label=y, softmax_output=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    elif loss_type == 'squared_error':
        for x, y in zip(validateX, validateY):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding loss w.r.t to a single datapoint
            loss_local += squared_error(label=y, softmax_output=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    average_loss = loss_local / float(len(validateX))
    acc = acc / float(len(validateX))
    return [average_loss, acc]


def augment_my_data(datapoints, labels, d, newSize):
    dataGenerator = ImageDataGenerator(rotation_range=15, shear_range=0.1, zoom_range=0.2, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
    new_data = []
    new_label = []
    datapoints = datapoints.reshape((d, 28, 28, 1))
    i = 0
    for (data, label) in dataGenerator.flow(datapoints, labels, batch_size=1):
        new_data.append(data.reshape(28, 28))
        new_label.append(label)
        i += 1
        if i > newSize:
            break

    return np.array(new_data), np.array(new_label), newSize


# 1 epoch = 1 pass over the data
def fit(datapoints, batch, epochs, labels, opt, loss_type, augment):
    n = len(network)  # number of layers
    d = len(datapoints)  # number of data points
    """This variable will be used to separate , training and validation set
        1) we take 10 % of the data as suggested in the question. -->int(d * .1)
        2) we also add any extra remaining data to validation set so that,
        training data is exactly divisible by batch size -->((d - int(d * .1)) % batch
    """
    border = d - ((d - int(d * .1)) % batch + int(d * .1))
    # separating the validation data
    validateX = datapoints[border:]
    validateY = labels[border:]
    # deleting copied datapoints
    datapoints = datapoints[:border]
    labels = labels[:border]
    # updating d
    d = border
    # augmenting my datapoints
    if augment is not None:
        (datapoints, labels, d) = augment_my_data(datapoints=datapoints, labels=labels, d=d, newSize=d + augment * batch)

    # is used to stochastically select our data.
    shuffler = np.arange(0, d)
    # creating simple gradient descent optimiser

    # loop for epoch iteration
    for k in range(epochs):
        # iteration for different starting point for epoch
        # shuffler at the start of each epoch
        np.random.shuffle(shuffler)
        for i in range(0, d - batch + 1, batch):
            clean = True
            # initiating loss for current epoch
            global loss
            loss = 0
            if isinstance(opt, NAG):
                opt.lookahead(network=network)
            # iterate over a batch
            for j in range(i, i + batch, 1):
                # creating a single data vector and normalising color values between 0 to 1
                x = datapoints[shuffler[j]].reshape(784, 1) / 255.0
                y = labels[shuffler[j]]
                forward_propagation(n, x)

                backward_propagation(n, x, y, number_of_datapoint=batch, loss_type=loss_type, clean=clean)
                clean = False

            opt.descent(network=network, gradient=gradient)

        # for wandb logging
        validation_result = validate(number_of_layer=n, validateX=validateX, validateY=validateY,
                                     loss_type=loss_type)
        training_result = validate(number_of_layer=n, validateX=datapoints,
                                   validateY=labels, loss_type=loss_type)

        # printing average loss.
        wandb.log({"val_accuracy": validation_result[1], 'val_loss': validation_result[0][0],
                   'train_accuracy': training_result[1], 'train_loss': training_result[0][0], 'epoch': k + 1})

        if np.isnan(validation_result[0])[0]:
            return


""" Adds a particular on top of previous layer , the layers are built in a incremental way.
    Context denotes the type of layer we have.Eg - Sigmoid or Tanh etc.
    Passing any number to input_dim it we counted as the first layer
 """


def add_layer(number_of_neurons, context, weight_init, input_dim=None):
    # Initialize an Empty Dictionary: layer
    layer = {}
    if weight_init == 'random':
        if input_dim is not None:
            layer['weight'] = np.random.rand(number_of_neurons, input_dim)
        else:
            # get number of neurons in the previous layer
            previous_lay_neuron_num = network[-1]['h'].shape[0]
            layer['weight'] = np.random.rand(number_of_neurons, previous_lay_neuron_num)

    elif weight_init == 'xavier':
        if input_dim is not None:
            layer['weight'] = np.random.normal(size=(number_of_neurons, input_dim))
            xavier = input_dim
        else:
            # get number of neurons in the previous layer
            previous_lay_neuron_num = network[-1]['h'].shape[0]
            layer['weight'] = np.random.normal(size=(number_of_neurons, previous_lay_neuron_num))
            xavier = previous_lay_neuron_num
        if context == 'relu':
            # relu has different optimal weight initialization.
            layer['weight'] = layer['weight'] * math.sqrt(2 / float(xavier))
        else:
            layer['weight'] = layer['weight'] * math.sqrt(1 / float(xavier))
    # initialise a 1-D array of size n with random samples from a uniform distribution over [0, 1).
    layer['bias'] = np.zeros((number_of_neurons, 1))
    # initialises a 2-D array of size [n*1] and type float with element having value as 1.
    layer['h'] = np.zeros((number_of_neurons, 1))
    layer['a'] = np.zeros((number_of_neurons, 1))
    layer['context'] = context
    network.append(layer)


"""master() is used to initialise all the learning parameters 
   in every layer and then start the training process"""


def master(batch, epochs, output_dim, activation, opt, layer_1, layer_2, layer_3, weight_init='xavier',loss_type='cross_entropy',
           augment=None):
    """initializing number of input features per datapoint as 784,
       since dataset consists of 28x28 pixel grayscale images
       :param augment: """
    n_features = 784
    global network
    global gradient
    global transient_gradient
    network = []
    gradient = []
    transient_gradient = []
    # adding layers
    add_layer(number_of_neurons=layer_1, context=activation, input_dim=784, weight_init=weight_init)
    # creating hidden layers
    add_layer(number_of_neurons=layer_2, context=activation, weight_init=weight_init)
    add_layer(number_of_neurons=layer_3, context=activation, weight_init=weight_init)
    add_layer(number_of_neurons=output_dim, context='softmax', weight_init=weight_init)

    """Copying the structure of network."""
    gradient = copy.deepcopy(network)
    transient_gradient = copy.deepcopy(network)
    fit(datapoints=trainX, labels=trainY, batch=batch, epochs=epochs, opt=opt,
        loss_type=loss_type,augment=augment)
    return network


def train():
    run = wandb.init()
    opti = None
    wandb.run.name = 'augmented_bs_' + str(run.config.batch_size) + '_act_' + run.config.activation + '_opt_' + str(
        run.config.optimiser) + '_ini_' + str(run.config.weight_init) + '_epoch' + str(run.config.epoch) + '_lr_' + str(
        round(run.config.learning_rate, 4) + str(run.config.loss))
    if run.config.optimiser == 'nag':
        opti = NAG(layers=4, eta=run.config.learning_rate, gamma=.90, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'rmsprop':
        opti = RMSProp(layers=4, eta=run.config.learning_rate, beta=.90, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'sgd':
        opti = SimpleGradientDescent(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'mom':
        opti = MomentumGradientDescent(layers=4, eta=run.config.learning_rate, gamma=.99,
                                       weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'adam':
        opti = ADAM(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'nadam':
        opti = NADAM(layers=4, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)

    master(epochs=run.config.epoch, batch=run.config.batch_size, output_dim=10,
           opt=opti, weight_init=run.config.weight_init, activation=run.config.activation, layer_1=run.config.layer_1,
           layer_3=run.config.layer_3, layer_2=run.config.layer_2, loss_type=run.config.loss, augment=100)