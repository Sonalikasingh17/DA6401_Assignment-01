# This is Assignment 01 of DA5401

The wandb report can be found in the following link:

https://wandb.ai/singhsonalika5-indian-institute-of-technology-madras/Fashion-MNIST-Images/reports/DA6401_Assigment-1-Report---VmlldzoxMTY5Nzc5Nw

The problem statement involves building and training a Neural Network from scratch using primarily Numpy package in Python.

The code base now has the following features:

Forward and backward propagation are hard coded using Matrix operations. The weights and biases are stored separately as dictionaries to go hand in hand with the notation used in class.
A neural network class to instantiate the neural network object for specified set of hyperparameters, namely the number of layers, hidden neurons, activation function, optimizer, weight decay,etc.
The optimisers, activations and their gradients are passed through dictionaries configured as attributed within the NeuralNetwork class.
Activation functions are defined separately in the activations.py file.


For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set  are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set.

Once the best configuration is identified with the help of wandb wither using  Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.

Code base structure
activations.py - contains all the activation functions and its derivatives.

optimizers.py - all the optimizers are seperately defined for the convenience.

train.py - dataset download, splitting and preprocessing along with training and hyper parameter sweep using wandb agent, the NeuralNetwork class is defined within this file.

Fashion_MNIST_dataset_images.py - data set download and plotting of sample images (Question 01)


Use the sweep configurations for wandb based on  Bayesian hyperparameter search can be configured in the following manner:

sweep_config = {
  "name": "Random Sweep", #(or) Bayesian Sweep (or) Grid search
  "method": "random", #(or) bayes (or) grid
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}
One can choose to select / modify/omit any of the hyperparameters above in the config dictionary.

Results:
The neural network implemented, the maximum test accuracy reported was 88.08% on the Fashion MNIST dataset. One of the model configuration chosen to be the best is as follows:

Number of Hidden Layers - 3
Number of Hidden Neurons - 128
L2 Regularisation - 0.0005
Activation - Tanh
Initialisation - Xavier
Optimiser - ADAM
Learning Rate - 0.0001
Batch size - 32
