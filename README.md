# This is Assignment 01 of DA5401

The wandb report can be found in the following link:

https://wandb.ai/singhsonalika5-indian-institute-of-technology-madras/Fashion-MNIST-Images/reports/DA6401_Assigment-1-Report---VmlldzoxMTY5Nzc5Nw

The Github repo can be found in the following link:
https://github.com/Sonalikasingh17/DA6401_Assignment-01/tree/main

The problem statement involves building and training a Neural Network from scratch using primarily Numpy package in Python.

### The code base now has the following features:

with the notation used in class. 

A neural network class is implemented to instantiate the neural network object for a specified set of hyperparameters, namely the number of layers, hidden neurons, activation function, optimizer, weight decay, etc.

The optimizers, activations, and their gradients are passed through dictionaries configured as attributes within the NeuralNetwork class.

Activation functions are defined separately in the `activations.py` file.

For the hyperparameter optimization stage, 10% of the randomly shuffled training dataset is kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54,000 images from the randomly shuffled training dataset.

Once the best configuration is identified with the help of wandb, either using Bayesian optimization or other methods, the full training dataset is used to train the best model configuration, and the test accuracy is calculated. The resulting confusion matrix is plotted thereafter.

### Code base structure
- `activations.py`: Contains all the activation functions and their derivatives.

- `optimizers.py`: All the optimizers are separately defined for convenience.

- `train.py`: Handles dataset download, splitting, preprocessing, training, and hyperparameter sweep using the wandb agent. The `NeuralNetwork` class is defined within this file.

- `Fashion_MNIST_dataset_images.py`: Handles dataset download and plotting of sample images (Question 01).

- `confusion_matrix_plot.py`: Used to plot the confusion matrix for the test dataset predictions.

The confusion matrix is generated using the `sklearn.metrics.confusion_matrix` function. The matrix is then visualized using matplotlib's heatmap functionality for better interpretability.

### Usage:
- Import the function from this file.
- Pass the true labels and predicted labels to generate and visualize the confusion matrix.
- Import the function from this file.
- Pass the true labels and predicted labels to generate and visualize the confusion matrix.


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
One can choose to select and modify any of the hyperparameters above in the config dictionary.

### Results:
The neural network implemented, the maximum test accuracy reported was 87.98% on the Fashion MNIST dataset. One of the model configuration chosen to be the best is as follows:

Number of Hidden Layers - 3
Number of Hidden Neurons - 128
L2 Regularisation - 0.0005
Activation - Tanh
Initialisation - Xavier
Optimiser - ADAM
Learning Rate - 0.0001
Batch size - 64
