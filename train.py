import numpy as np
import math
import wandb
from optimiser import *
from loss import *
from gradient import *
from activation import *
from keras.datasets import fashion_mnist


""" get training and sting vectors
    Number of Training Images = 60000
    Number of Testing Images = 10000 """

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_test.shape[0]
x_train.shape[0]
y_test.shape[0]
y_train.shape[0]

# Here,network is a list of all the learning parameters in every layer and 
# layer_gradient is its copy
network = []
layer_gradient = []

delta = []  # store layer_gradient w.r.t a single datapoint
# Stores the cumulative loss for each update step (timestep = 1 parameter update).
total_loss = 0


def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        network[i]['h'] = activation_function(network[i]['a'], context=network[i]['context'])


def backward_propagation(number_of_layers, x, y, number_of_datapoint, loss_type, clean=False):
    delta[number_of_layers - 1]['h'] = output_grad(network[number_of_layers - 1]['h'], y,
                                                                loss_type=loss_type)
    delta[number_of_layers - 1]['a'] = last_grad(network[number_of_layers - 1]['h'], y)
    for i in range(number_of_layers - 2, -1, -1):
        delta[i]['h'] = h_grad(network=network, delta=delta, layer=i)
        delta[i]['a'] = a_grad(network=network, delta=delta, layer=i)
    for i in range(number_of_layers - 1, -1, -1):
        delta[i]['weight'] = w_grad(network=network, delta=delta, layer=i, x=x)
        delta[i]['bias'] = layer_gradient[i]['a']
    if clean:
        layer_gradient[number_of_layers - 1]['h'] = delta[number_of_layers - 1]['h'] / float(number_of_datapoint)
        layer_gradient[number_of_layers - 1]['a'] = delta[number_of_layers - 1]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            layer_gradient[i]['h'] = delta[i]['h'] / float(number_of_datapoint)
            layer_gradient[i]['a'] = delta[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            layer_gradient[i]['weight'] = delta[i]['weight'] / float(number_of_datapoint)
            layer_gradient[i]['bias'] = delta[i]['bias'] / float(number_of_datapoint)
    else:

        layer_gradient[number_of_layers - 1]['h'] += delta[number_of_layers - 1]['h'] / float(
            number_of_datapoint)
        layer_gradient[number_of_layers - 1]['a'] += delta[number_of_layers - 1]['a'] / float(
            number_of_datapoint)
        for i in range(number_of_layers - 2, -1, -1):
            layer_gradient[i]['h'] += delta[i]['h'] / float(number_of_datapoint)
            layer_gradient[i]['a'] += delta[i]['a'] / float(number_of_datapoint)
        for i in range(number_of_layers - 1, -1, -1):
            layer_gradient[i]['weight'] += delta[i]['weight'] / float(number_of_datapoint)
            layer_gradient[i]['bias'] += delta[i]['bias'] / float(number_of_datapoint)


# Evaluating the model performance on validation data to assist with hyperparameter tuning.

def evaluate_model(number_of_layer, x_val, y_val, loss_type):
    loss_local = 0
    acc = 0
    if loss_type == 'cross_entropy':
        for x, y in zip(x_val, y_val):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding total_loss w.r.t to a single datapoint
            loss_local += cross_entropy(label=y, y_pred=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    elif loss_type == 'squared_error':
        for x, y in zip(x_val, y_val):
            forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
            # adding total_loss w.r.t to a single datapoint
            loss_local += squared_error(label=y, y_pred=network[number_of_layer - 1]['h'])
            max_prob = np.argmax(network[number_of_layer - 1]['h'])
            if max_prob == y:
                acc += 1
    average_loss = loss_local / float(len(x_val))
    acc = acc / float(len(x_val))
    return [average_loss, acc]


# 1 epoch = 1 pass over the data

def train_model(datapoints, batch, epochs, labels, opt, loss_type):
    n = len(network)  # number of layers
    d = len(datapoints)  # number of data points

    """ This is used to split the dataset into training and validation sets.  
    1) 10% of the data is allocated for validation: int(d * 0.1).  
    2) Any extra remaining data is added to the validation set to ensure that  
       the training set size is perfectly divisible by the batch size:  
       ((d - int(d * 0.1)) % batch). """  

    border = d - ((d - int(d * .1)) % batch + int(d * .1))
    # separating the validation data
    x_val = datapoints[border:]
    y_val = labels[border:]
    # deleting copied datapoints
    datapoints = datapoints[:border]
    labels = labels[:border]
    # updating d
    d = border
    
    # for stochastically select our data.
    shuffler = np.arange(0, d)
    # creating simple layer_gradient descent optimiser

    # loop for epoch iteration
    for k in range(epochs):
        # iteration for different starting point for epoch
        # shuffler at the start of each epoch
        np.random.shuffle(shuffler)
        for i in range(0, d - batch + 1, batch):
            clean = True
            # initiating total_loss for current epoch
            global total_loss
            total_loss = 0
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

            opt.descent(network=network, layer_gradient=layer_gradient)

        # for wandb logging
        validation_result = evaluate_model(number_of_layer=n, x_val=x_val, y_val=y_val,
                                     loss_type=loss_type)
        training_result = evaluate_model(number_of_layer=n, x_val=datapoints,
                                   y_val=labels, loss_type=loss_type)

        # printing average total_loss.
        wandb.log({"val_accuracy": validation_result[1], 'val_loss': validation_result[0][0],
                   'train_accuracy': training_result[1], 'train_loss': training_result[0][0], 'epoch': k + 1})

        if np.isnan(validation_result[0])[0]:
            return


""" Adds a new layer on top of the existing ones, constructing the network incrementally.  
    The 'context' parameter specifies the activation function (e.g., Sigmoid, Tanh, ReLU).  
    If 'input_dim' is provided, the layer is treated as the first input layer. """  


def add_network_layer(number_of_neurons, context, weight_init, input_dim=None):
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
    
    """ Initializes the number of input features per data point as 784,  
    since each image in the dataset is a 28x28 grayscale image (28 × 28 = 784 pixels).  
    :Parameter for data augmentation (if applicable). """  

    # n_features = 784
    global network
    global layer_gradient
    global delta
    network = []
    layer_gradient = []
    delta = []
    # adding layers
    add_network_layer(number_of_neurons=layer_1, context=activation, input_dim=784, weight_init=weight_init)
    # creating hidden layers
    add_network_layer(number_of_neurons=layer_2, context=activation, weight_init=weight_init)
    add_network_layer(number_of_neurons=layer_3, context=activation, weight_init=weight_init)
    add_network_layer(number_of_neurons=output_dim, context='sigmoid', weight_init=weight_init)

    """Duplicating the model structure to store gradients during backpropagation."""

    layer_gradient = {key: np.copy(value) for key, value in network.items()}
    delta = {key: np.copy(value) for key, value in network.items()}
    train_model(datapoints=x_train, labels=y_train, batch=batch, epochs=epochs, opt=opt,
        loss_type=loss_type,augment=augment)
    return network


def train():
    run = wandb.init(project="Fashion-MNIST-Images")
    opti = None
    wandb.run.name = 'bs_' + str(run.config.batch_size) + '_act_' + run.config.activation + '_opt_' + str(
        run.config.optimiser) + '_ini_' + str(run.config.weight_init) + '_epoch' + str(run.config.epoch) + '_lr_' + str(
        round(run.config.learning_rate, 4) + str(run.config.total_loss))
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
           layer_3=run.config.layer_3, layer_2=run.config.layer_2, loss_type=run.config.total_loss, augment=100)