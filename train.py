import numpy as np
import wandb
import time
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# Activation functions

def sigmoid(x):
    return 1.0 / (1 + np.exp(-(x)))
def sigmoid_derivative(x):
    return  (1.0 / (1 + np.exp(-(x))))*(1 -  1.0 / (1 + np.exp(-(x))))

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0.001, x) 
def relu_derivative(x):
    return (x>0)*1 + (x<0)*0.001 


class NeuralNetwork:
    def __init__(self, hidden_layers, hidden_neurons, train_images, train_labels, num_train, val_images,val_labels,num_val,test_images, test_labels, num_test,optimizer,batch_size,weight_decay,learning_rate,
    max_epochs,activation,initializer,loss_function):
        
        self.num_classes = np.max(train_labels) + 1
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.output_size = self.num_classes
        self.img_height = train_images.shape[1]
        self.img_width = train_images.shape[2]
        self.input_size = self.img_height * self.img_width

        self.layer_structure = ([self.input_size]+ hidden_layers * [hidden_neurons]+ [self.output_size])
    

        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        
        self.train_data = np.transpose(
            train_images.reshape(train_images.shape[0], -1)) / 255
        self.test_data = np.transpose(
            test_images.reshape(test_images.shape[0], -1)) / 255
        self.val_data = np.transpose(
            val_images.reshape(val_images.shape[0], -1)) / 255
        
        self.train_labels = self.one_hot_encode(train_labels)
        self.val_labels = self.one_hot_encode(val_labels)
        self.test_labels = self.one_hot_encode(test_labels)

        self.activation_functions = {"SIGMOID": sigmoid, "TANH": tanh, "RELU": relu}
        self.derivative_activations = {"SIGMOID": sigmoid_derivative,
                                       "TANH": tanh_derivative,
                                       "RELU": relu_derivative}


        self.initializers = {"XAVIER": self.xavier_initializer,"RANDOM": self.random_initializer}

        self.optimizers = {"SGD": self.sgd,    "MGD": self.mgd,
            "NGD": self.ngd,  "RMSPROP": self.rmsProp, "ADAM": self.adam}
        
    
        self.activation_func = self.activation_functions[activation]
        self.derivative_activations = self.derivative_activations[activation]
    
        self.optimizer = self.optimizers[optimizer]
        self.initializer_func = self.initializers[initializer]
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.weights, self.biases = self.initialize_network(self.layer_structure)

    def one_hot_encode(self, labels):
        encoded_labels = np.zeros((self.num_classes, labels.shape[0]))
        for i in range(labels.shape[0]):
            encoded_labels[int(labels[i]), i] = 1.0
        return encoded_labels
     
    #  Loss functions
    def mean_squared_error(self, true_labels, predicted_labels):
        return np.mean((true_labels - predicted_labels) ** 2)

    def cross_entropy_loss(self, true_labels, predicted_labels):
        return np.mean([-true_labels[i] * np.log(predicted_labels[i]) for i in range(len(predicted_labels))])

    def l2_regularisation_loss(self, weight_decay):
        return weight_decay * np.sum([np.linalg.norm(self.weights[str(i + 1)]) ** 2 for i in range(len(self.weights))])
    

    def compute_accuracy(self, true_labels, predicted_labels, data_size):
        true_class_labels = [np.argmax(true_labels[:, i]) for i in range(data_size)]
        predicted_class_labels = [np.argmax(predicted_labels[:, i]) for i in range(data_size)]
        correct_predictions = sum(1 for i in range(data_size) if true_class_labels[i] == predicted_class_labels[i])
        return correct_predictions / data_size, true_class_labels, predicted_class_labels

    def xavier_initializer(self, size):
        in_dim, out_dim = size[1], size[0]
        std_dev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal(0, std_dev, size=(out_dim, in_dim))

    def random_initializer(self, size):
        in_dim, out_dim = size[1], size[0]
        return np.random.normal(0, 1, size=(out_dim, in_dim))

        
    def initialize_network(self, layer_structure):
        weight_matrices = {}
        bias_vectors = {}
        total_layers = len(layer_structure)
        for i in range(0, total_layers - 1):
            weight_matrix = self.initializer_func(size=[layer_structure[i + 1], layer_structure[i]])
            bias_vector = np.zeros((layer_structure[i + 1], 1))
            weight_matrices[str(i + 1)] = weight_matrix
            bias_vectors[str(i + 1)] = bias_vector
        return weight_matrices, bias_vectors

    def forward_propagation(self, input_batch, weight_matrices, bias_vectors):
        """
        Returns the neural network output given input data, weights, and biases.
        Arguments:
                  input_batch - input matrix
                  weight_matrices - Weight matrices
                  bias_vectors - Bias vectors 
        """
        num_layers = len(weight_matrices) + 1
        activation_outputs = {}
        pre_activations = {}
        activation_outputs["0"] = input_batch
        pre_activations["0"] = input_batch
        
        for i in range(0, num_layers - 2):
            if i == 0:
                W = weight_matrices[str(i + 1)]
                b = bias_vectors[str(i + 1)]
                pre_activations[str(i + 1)] = np.add(np.matmul(W, input_batch), b)
                activation_outputs[str(i + 1)] = self.activation_func(pre_activations[str(i + 1)])
            else:
                W = weight_matrices[str(i + 1)]
                b = bias_vectors[str(i + 1)]
                pre_activations[str(i + 1)] = np.add(np.matmul(W, activation_outputs[str(i)]), b)
                activation_outputs[str(i + 1)] = self.activation_func(pre_activations[str(i + 1)])

        W = weight_matrices[str(num_layers - 1)]
        b = bias_vectors[str(num_layers - 1)]
        pre_activations[str(num_layers - 1)] = np.add(np.matmul(W, activation_outputs[str(num_layers - 2)]), b)
        final_output = sigmoid(pre_activations[str(num_layers - 1)])
        activation_outputs[str(num_layers - 1)] = final_output
        return final_output, activation_outputs, pre_activations
    
    def back_propagation(self, predicted_output, activation_outputs, pre_activations, true_output, weight_decay=0):
        '''
        Performs backpropagation to compute gradients of weights and biases.
    
         Arguments:
           predicted_output - Output of the neural network
           activation_outputs - Dictionary of activation outputs from forward propagation
           pre_activations - Dictionary of pre-activation values
           true_output - True labels
           weight_decay - Regularization parameter (default: 0)
        
        Returns:
           weight_gradients - Gradients for weight matrices
           bias_gradients - Gradients for bias vectors 
        '''
        
        alpha = weight_decay
        weight_gradients= {}
        bias_gradients = {}
        num_layers = len(self.layer_structure)
    
    # Compute gradient of the output layer
        activation_gradients = {}
        if self.loss_function == "CROSS":
           activation_gradients[str(num_layers - 1)] = -(true_output - predicted_output)
        elif self.loss_function == "MSE":
            activation_gradients[str(num_layers - 1)] = np.multiply(
            2 * (predicted_output - true_output), np.multiply(predicted_output, (1 - predicted_output))
        )

    # Backpropagate through the layers
        for i in range(num_layers - 2, -1, -1):
            if alpha != 0:
              weight_gradients[str(i + 1)] = (
                np.outer(activation_gradients[str(i + 1)], activation_outputs[str(i)])
                + alpha * self.weights[str(i + 1)]
            )
            else:
              weight_gradients[str(i + 1)] = np.outer(activation_gradients[str(i + 1)], activation_outputs[str(i)])

            bias_gradients[str(i + 1)] = activation_gradients[str(i + 1)]

            if i != 0:
               hidden_gradient = np.matmul(self.weights[str(i + 1)].T, activation_gradients[str(i + 1)])

               activation_gradients[str(i)] = np.multiply(hidden_gradient, self.derivative_activations(pre_activations[str(i)]))

            else:
               hidden_gradient = np.matmul(self.weights[str(i + 1)].T, activation_gradients[str(i + 1)])
               activation_gradients[str(i)] = np.multiply(hidden_gradient, pre_activations[str(i)])

        return weight_gradients, bias_gradients
    
    
    # Predicts the output for a given input dataset using forward propagation.
    def predict(self, input_batch, length_dataset):
        predictions = []        
    
        for i in range(length_dataset):
            final_output, activation_outputs, pre_activations = self.forward_propagation(
                input_batch[:, i].reshape(self.input_size, 1),
                self.weights,
                self.biases,
        )

            predictions.append(final_output.reshape(self.num_classes,))
    
        predictions = np.array(predictions).transpose()
        return predictions


    def sgd(self, epochs, length_dataset, learning_rate, weight_decay=0):

        training_loss = []
        training_accuracy = []
        validation_accuracy = []
    
        num_layers = len(self.layer_structure)

        X_train = self.train_data[:, :length_dataset]
        Y_train = self.train_labels[:, :length_dataset]

        for epoch in range(epochs):
            start_time = time.time()
        
            indices = np.arange(length_dataset)
            np.random.shuffle(indices)
            X_train = X_train[:, indices].reshape(self.input_size, length_dataset)
            Y_train = Y_train[:, indices].reshape(self.num_classes, length_dataset)
        
            batch_loss = []
        
            weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
            bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            for i in range(length_dataset):
                output, activation_outputs, pre_activations = self.forward_propagation(
                    X_train[:, i].reshape(self.input_size, 1),
                    self.weights,
                    self.biases,
            )
                weight_gradients, bias_gradients = self.back_propagation(
                    output, activation_outputs, pre_activations, Y_train[:, i].reshape(self.num_classes, 1), weight_decay
            )

                for l in range(num_layers - 1):
                    weight_updates[str(l + 1)] = weight_gradients[str(l + 1)]
                    bias_updates[str(l + 1)] = bias_gradients[str(l + 1)]

                if self.loss_function == "MSE":
                    batch_loss.append(
                       self.mean_squared_error(Y_train[:, i].reshape(self.num_classes, 1), output)
                         + self.l2_regularisation_loss(weight_decay)
                )
                elif self.loss_function == "CROSS":
                    batch_loss.append(
                    self.cross_entropy_loss(Y_train[:, i].reshape(self.num_classes, 1), output)
                    + self.l2_regularisation_loss(weight_decay)
                )

                self.weights = {str(l + 1): (self.weights[str(l + 1)] - learning_rate * weight_updates[str(l + 1)])
                            for l in range(len(self.weights))}
                self.biases = {str(l + 1): (self.biases[str(l + 1)] - learning_rate * bias_updates[str(l + 1)])
                           for l in range(len(self.biases))}

            elapsed_time = time.time() - start_time
        
            predictions = self.predict(self.train_data, self.num_train)
        
            training_loss.append(np.mean(batch_loss))
            training_accuracy.append(self.compute_accuracy(Y_train, predictions, length_dataset)[0])
            validation_accuracy.append(self.compute_accuracy(self.num_test, self.predict(self.num_train, self.num_val), self.num_val)[0])
        
            print(
                "Epoch: %d, Loss: %.3e, Training Accuracy: %.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                % (
                epoch,training_loss[epoch],training_accuracy[epoch],validation_accuracy[epoch],elapsed_time,learning_rate,
            )
        )

            wandb.log({'loss': np.mean(batch_loss),'training_accuracy': training_accuracy[epoch],'validation_accuracy': validation_accuracy[epoch],'epoch': epoch})

        return training_loss, training_accuracy, validation_accuracy, predictions

   
    def mgd(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):

        MOMENTUM = 0.9
        training_loss = []
        training_accuracy = []
        validation_accuracy = []

        num_layers = len(self.layer_structure)

        X_train = self.train_data[:, :length_dataset]
        Y_train = self.train_labels[:, :length_dataset]

        prev_velocity_w = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
        prev_velocity_b = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

        num_points_seen = 0

        for epoch in range(epochs):
            start_time = time.time()

            indices = np.arange(length_dataset)
            np.random.shuffle(indices)
            X_train = X_train[:, indices].reshape(self.input_size, length_dataset)
            Y_train = Y_train[:, indices].reshape(self.num_classes, length_dataset)

            batch_loss = []

            weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
            bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            for i in range(length_dataset):
                output, activation_outputs, pre_activations = self.forward_propagation(
                    X_train[:, i].reshape(self.input_size, 1),
                    self.weights,
                    self.biases,
            )
                weight_gradients, bias_gradients = self.back_propagation(
                    output, activation_outputs, pre_activations, Y_train[:, i].reshape(self.num_classes, 1)
            )

                for l in range(num_layers - 1):
                    weight_updates[str(l + 1)] += weight_gradients[str(l + 1)]
                    bias_updates[str(l + 1)] += bias_gradients[str(l + 1)]

                if self.loss_function == "MSE":
                    batch_loss.append(
                        self.mean_squared_error(Y_train[:, i].reshape(self.num_classes, 1), output)
                        + self.l2_regularisation_loss(weight_decay)
                )
                elif self.loss_function == "CROSS":
                    batch_loss.append(
                        self.cross_entropy_loss(Y_train[:, i].reshape(self.num_classes, 1), output)
                        + self.l2_regularisation_loss(weight_decay)
                )

                num_points_seen += 1

                if num_points_seen % batch_size == 0:
                    velocity_w = {
                       str(l + 1): MOMENTUM * prev_velocity_w[str(l + 1)] + learning_rate * weight_updates[str(l + 1)] / batch_size
                       for l in range(num_layers - 1)
                }
                    velocity_b = {
                       str(l + 1): MOMENTUM * prev_velocity_b[str(l + 1)] + learning_rate * bias_updates[str(l + 1)] / batch_size
                        for l in range(num_layers - 1)
                }

                    self.weights = {str(l + 1): self.weights[str(l + 1)] - velocity_w[str(l + 1)] for l in range(num_layers - 1)}
                    self.biases = {str(l + 1): self.biases[str(l + 1)] - velocity_b[str(l + 1)] for l in range(num_layers - 1)}

                    prev_velocity_w = velocity_w
                    prev_velocity_b = velocity_b

                    # Reset batch gradients
                    weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
                    bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            elapsed_time = time.time() - start_time

            predictions = self.predict(self.train_data, self.num_train)

            training_loss.append(np.mean(batch_loss))
            training_accuracy.append(self.compute_accuracy(Y_train, predictions, length_dataset)[0])
            validation_accuracy.append(self.compute_accuracy(self.test_labels, self.predict(self.test_data, self.num_test), self.num_test)[0])

            print(
            "Epoch: %d, Loss: %.3e, Training Accuracy: %.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
            % (
                epoch,training_loss[epoch],training_accuracy[epoch],
                validation_accuracy[epoch],elapsed_time,learning_rate,
            )
        )

            wandb.log({'loss': np.mean(batch_loss),'training_accuracy': training_accuracy[epoch],'validation_accuracy': validation_accuracy[epoch],'epoch': epoch
        })

        return training_loss, training_accuracy, validation_accuracy, predictions
    
    def ngd(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
    
        GAMMA = 0.9
    
        X_train = self.train_data[:, :length_dataset]
        Y_train = self.train_labels[:, :length_dataset]

        training_loss = []
        training_accuracy = []
        validation_accuracy = []
    
        num_layers = len(self.layer_structure)
    
        prev_velocity_w = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
        prev_velocity_b = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}
    
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
        
            indices = np.arange(length_dataset)
            np.random.shuffle(indices)
            X_train = X_train[:, indices].reshape(self.input_size, length_dataset)
            Y_train = Y_train[:, indices].reshape(self.num_classes, length_dataset)

            batch_loss = []
        
            weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
            bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}
        
            velocity_w = {str(l + 1): GAMMA * prev_velocity_w[str(l + 1)] for l in range(num_layers - 1)}
            velocity_b = {str(l + 1): GAMMA * prev_velocity_b[str(l + 1)] for l in range(num_layers - 1)}
        
            for i in range(length_dataset):
                winter = {str(l + 1): self.weights[str(l + 1)] - velocity_w[str(l + 1)] for l in range(num_layers - 1)}
                binter = {str(l + 1): self.biases[str(l + 1)] - velocity_b[str(l + 1)] for l in range(num_layers - 1)}
            
                output, activation_outputs, pre_activations = self.forward_propagation(
                    X_train[:, i].reshape(self.input_size, 1), winter, binter)
                weight_gradients, bias_gradients = self.back_propagation(
                    output, activation_outputs, pre_activations, Y_train[:, i].reshape(self.num_classes, 1))
            
                for l in range(num_layers - 1):
                    weight_updates[str(l + 1)] += weight_gradients[str(l + 1)]
                    bias_updates[str(l + 1)] += bias_gradients[str(l + 1)]
            
                if self.loss_function == "MSE":
                    batch_loss.append(
                    self.mean_squared_error(Y_train[:, i].reshape(self.num_classes, 1), output)
                    + self.l2_regularisation_loss(weight_decay)
                )
                elif self.loss_function == "CROSS":
                    batch_loss.append(
                    self.cross_entropy_loss(Y_train[:, i].reshape(self.num_classes, 1), output)
                    + self.l2_regularisation_loss(weight_decay)
                )
            
                num_points_seen += 1
            
                if num_points_seen % batch_size == 0:
                    velocity_w = {
                        str(l + 1): GAMMA * prev_velocity_w[str(l + 1)] + learning_rate * weight_updates[str(l + 1)] / batch_size
                        for l in range(num_layers - 1)
                }
                velocity_b = {
                    str(l + 1): GAMMA * prev_velocity_b[str(l + 1)] + learning_rate * bias_updates[str(l + 1)] / batch_size
                    for l in range(num_layers - 1)
                }
                
                self.weights = {str(l + 1): self.weights[str(l + 1)] - velocity_w[str(l + 1)] for l in range(num_layers - 1)}
                self.biases = {str(l + 1): self.biases[str(l + 1)] - velocity_b[str(l + 1)] for l in range(num_layers - 1)}
                
                prev_velocity_w = velocity_w
                prev_velocity_b = velocity_b
                
                weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
                bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}
        
            elapsed_time = time.time() - start_time
            predictions = self.predict(self.train_data, self.num_train)
        
            training_loss.append(np.mean(batch_loss))
            training_accuracy.append(self.compute_accuracy(Y_train, predictions, length_dataset)[0])
            validation_accuracy.append(self.compute_accuracy(self.test_labels, self.predict(self.     test_data, self.num_test), self.num_test)[0])
        
            print(
            "Epoch: %d, Loss: %.3e, Training Accuracy: %.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
            % (
                epoch,training_loss[epoch],training_accuracy[epoch],
                validation_accuracy[epoch],elapsed_time,learning_rate
            )
        )
        
            wandb.log({'loss': np.mean(batch_loss),'training_accuracy': training_accuracy[epoch],'validation_accuracy': validation_accuracy[epoch],'epoch': epoch
        })
        
        return training_loss, training_accuracy, validation_accuracy, predictions

    
    def rmsProp(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0): 

        EPSILON = 1e-8
        BETA = 0.9

        X_train = self.train_data[:, :length_dataset]
        Y_train = self.train_labels[:, :length_dataset]

        training_loss = []
        training_accuracy = []
        validation_accuracy = []
    
        num_layers = len(self.layer_structure)

        v_w = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
        v_b = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()

            indices = np.arange(length_dataset)
            np.random.shuffle(indices)
            X_train = X_train[:, indices].reshape(self.input_size, length_dataset)
            Y_train = Y_train[:, indices].reshape(self.num_classes, length_dataset)

            batch_loss = []

            weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
            bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            for i in range(length_dataset):
                output, activation_outputs, pre_activations = self.forward_propagation(
                    X_train[:, i].reshape(self.input_size, 1), self.weights, self.biases)
                
                weight_gradients, bias_gradients = self.back_propagation(
                    output, activation_outputs, pre_activations, Y_train[:, i].reshape(self.num_classes, 1))

                for l in range(num_layers - 1):
                    weight_updates[str(l + 1)] += weight_gradients[str(l + 1)]
                    bias_updates[str(l + 1)] += bias_gradients[str(l + 1)]

                if self.loss_function == "MSE":
                    batch_loss.append(
                    self.mean_squared_error(Y_train[:, i].reshape(self.num_classes, 1), output)
                    + self.l2_regularisation_loss(weight_decay)
                )
                elif self.loss_function == "CROSS":
                    batch_loss.append(
                    self.cross_entropy_loss(Y_train[:, i].reshape(self.num_classes, 1), output)
                    + self.l2_regularisation_loss(weight_decay)
                )

                num_points_seen += 1

                if num_points_seen % batch_size == 0:
                    v_w = {
                        str(l + 1): BETA * v_w[str(l + 1)] + (1 - BETA) * (weight_updates[str(l + 1)] / batch_size) ** 2
                        for l in range(num_layers - 1)
                }
                    v_b = {
                        str(l + 1): BETA * v_b[str(l + 1)] + (1 - BETA) * (bias_updates[str(l + 1)] / batch_size) ** 2
                        for l in range(num_layers - 1)
                }

                    self.weights = {
                        str(l + 1): self.weights[str(l + 1)] - (learning_rate / np.sqrt(v_w[str(l + 1)] + EPSILON)) * weight_updates[str(l + 1)] / batch_size
                        for l in range(num_layers - 1)
                }
                    self.biases = {
                        str(l + 1): self.biases[str(l + 1)] - (learning_rate / np.sqrt(v_b[str(l + 1)] + EPSILON)) * bias_updates[str(l + 1)] / batch_size
                        for l in range(num_layers - 1)
                }

                    weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
                    bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            elapsed_time = time.time() - start_time
            predictions = self.predict(self.train_data, self.num_train)

            training_loss.append(np.mean(batch_loss))
            training_accuracy.append(self.compute_accuracy(Y_train, predictions, length_dataset)[0])
            validation_accuracy.append(self.compute_accuracy(self.test_labels, self.predict(self.test_data, self.num_test), self.num_test)[0])

            print(
            "Epoch: %d, Loss: %.3e, Training Accuracy: %.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
            % (
                epoch,training_loss[epoch],training_accuracy[epoch],
                validation_accuracy[epoch],elapsed_time,learning_rate
            )
                )

            wandb.log({'loss': np.mean(batch_loss),'training_accuracy': training_accuracy[epoch],'validation_accuracy': validation_accuracy[epoch],'epoch': epoch
               })

        return training_loss, training_accuracy, validation_accuracy, predictions
    
    def adam(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
       
        EPSILON = 1e-8
        BETA1, BETA2 = 0.9, 0.99

        X_train = self.train_data[:, :length_dataset]
        Y_train = self.train_labels[:, :length_dataset]

        training_loss = []
        training_accuracy = []
        validation_accuracy = []
        
        num_layers = len(self.layer_structure)

        # Initialize first and second moment estimates
        m_w = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
        m_b = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

        v_w = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
        v_b = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()

            indices = np.arange(length_dataset)
            np.random.shuffle(indices)
            X_train = X_train[:, indices].reshape(self.input_size, length_dataset)
            Y_train = Y_train[:, indices].reshape(self.num_classes, length_dataset)

            batch_loss = []

            weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
            bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            for i in range(length_dataset):
                output, activation_outputs, pre_activations = self.forward_propagation(
                    X_train[:, i].reshape(self.input_size, 1), self.weights, self.biases)
                weight_gradients, bias_gradients = self.back_propagation(
                    output, activation_outputs, pre_activations, Y_train[:, i].reshape(self.num_classes, 1))

                for l in range(num_layers - 1):
                    weight_updates[str(l + 1)] += weight_gradients[str(l + 1)]
                    bias_updates[str(l + 1)] += bias_gradients[str(l + 1)]

                if self.loss_function == "MSE":
                    batch_loss.append(
                        self.mean_squared_error(Y_train[:, i].reshape(self.num_classes, 1), output)
                        + self.l2_regularisation_loss(weight_decay)
                    )
                elif self.loss_function == "CROSS":
                    batch_loss.append(
                        self.cross_entropy_loss(Y_train[:, i].reshape(self.num_classes, 1), output)
                        + self.l2_regularisation_loss(weight_decay)
                    )

                num_points_seen += 1

                if num_points_seen % batch_size == 0:
                    # Compute biased first and second moment estimates
                    m_w = {str(l + 1): BETA1 * m_w[str(l + 1)] + (1 - BETA1) * (weight_updates[str(l + 1)] / batch_size)
                        for l in range(num_layers - 1)}
                    m_b = {str(l + 1): BETA1 * m_b[str(l + 1)] + (1 - BETA1) * (bias_updates[str(l + 1)] / batch_size)
                        for l in range(num_layers - 1)}

                    v_w = {str(l + 1): BETA2 * v_w[str(l + 1)] + (1 - BETA2) * (weight_updates[str(l + 1)] / batch_size) ** 2
                        for l in range(num_layers - 1)}
                    v_b = {str(l + 1): BETA2 * v_b[str(l + 1)] + (1 - BETA2) * (bias_updates[str(l + 1)] / batch_size) ** 2
                        for l in range(num_layers - 1)}

                    # Bias correction
                    m_w_hat = {str(l + 1): m_w[str(l + 1)] / (1 - BETA1 ** (epoch + 1)) for l in range(num_layers - 1)}
                    m_b_hat = {str(l + 1): m_b[str(l + 1)] / (1 - BETA1 ** (epoch + 1)) for l in range(num_layers - 1)}

                    v_w_hat = {str(l + 1): v_w[str(l + 1)] / (1 - BETA2 ** (epoch + 1)) for l in range(num_layers - 1)}
                    v_b_hat = {str(l + 1): v_b[str(l + 1)] / (1 - BETA2 ** (epoch + 1)) for l in range(num_layers - 1)}

                    # Update weights and biases
                    self.weights = {
                        str(l + 1): self.weights[str(l + 1)] - (learning_rate / (np.sqrt(v_w_hat[str(l + 1)] + EPSILON))) * m_w_hat[str(l + 1)]
                        for l in range(num_layers - 1)
                    }
                    self.biases = {
                        str(l + 1): self.biases[str(l + 1)] - (learning_rate / (np.sqrt(v_b_hat[str(l + 1)] + EPSILON))) * m_b_hat[str(l + 1)]
                        for l in range(num_layers - 1)
                    }

                    weight_updates = {str(l + 1): np.zeros_like(self.weights[str(l + 1)]) for l in range(num_layers - 1)}
                    bias_updates = {str(l + 1): np.zeros_like(self.biases[str(l + 1)]) for l in range(num_layers - 1)}

            elapsed_time = time.time() - start_time
            predictions = self.predict(self.train_data, self.num_train)

            training_loss.append(np.mean(batch_loss))
            training_accuracy.append(self.compute_accuracy(Y_train, predictions, length_dataset)[0])
            validation_accuracy.append(self.compute_accuracy(self.test_labels, self.predict(self.test_data, self.num_test), self.num_test)[0])

            print(
                "Epoch: %d, Loss: %.3e, Training Accuracy: %.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                % (
                    epoch,training_loss[epoch],training_accuracy[epoch],validation_accuracy[epoch],elapsed_time,learning_rate,
                )
            )

            wandb.log({'loss': np.mean(batch_loss),'training_accuracy': training_accuracy[epoch],'validation_accuracy': validation_accuracy[epoch],'epoch': epoch
            })

        return training_loss, training_accuracy, validation_accuracy, predictions
    

# Load the datasets  
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define dataset sizes
total_train_samples = train_labels.shape[0]
train_size = int(0.9 * total_train_samples)  # 90% for training
validation_size = total_train_samples - train_size  # 10% for validation
test_size = test_labels.shape[0]

# Shuffle indices to randomize the dataset
train_indices = np.random.permutation(total_train_samples)

# Split dataset into training, validation, and test sets
shuffled_train_images = train_images[train_indices]
shuffled_train_labels = train_labels[train_indices]

X_train = shuffled_train_images[:train_size, :]
Y_train = shuffled_train_labels[:train_size]

X_valid = shuffled_train_images[train_size:, :]
Y_valid = shuffled_train_labels[train_size:]

X_test = test_images
Y_test = test_labels

# Define sweep configuration for hyperparameter tuning
sweep_config = {
    "name": "Bayes_Hyperparam_Tuning",
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "values": [5, 10]
        },"init_method": {
            "values": ["RANDOM", "XAVIER"]
        },"layers": {
            "values": [2, 3, 4]
        },"hidden_neurons": {
            "values": [32, 64, 128]
        },"activation_func": {
            "values": ['TANH', 'SIGMOID', 'RELU']
        },"learning_rate": {
            "values": [0.001, 0.0001]
        },"weight_decay": {
            "values": [0, 0.0005, 0.5]
        },"optimizer": {
            "values": ["SGD", "MGD", "NGD", "RMSPROP", "ADAM"]
        },"batch_size": {
            "values": [16, 32, 64]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project='Fashion_MNIST_Images', entity='singhsonalika5-indian-institute-of-technology-madras')


def train():    
        config_defaults = dict(
            max_epochs=5,hidden_layers=3,hidden_neurons=32,weight_decay=0,learning_rate=1e-3,optimizer="MGD",batch_size=16,activation="TANH",
            initializer="XAVIER",loss_function="CROSS",
        )
        
        wandb.init(config=config_defaults)
    
        wandb.run.name = "hl_" + str(wandb.config.hidden_layers) + "_hn_" + str(wandb.config.hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    
        CONFIG = wandb.config

        NN = NeuralNetwork(
        hidden_layers=CONFIG.hidden_layers,  hidden_neurons=CONFIG.hidden_neurons,
        train_images=train_images,           train_labels=train_labels,
        val_images=X_valid,                  val_labels=Y_valid,
        test_images=X_test,                  test_labels=Y_test,
        num_test=test_size,                  num_train=train_size,            
        optimizer=CONFIG.optimizer,          batch_size=CONFIG.batch_size,
        weight_decay=CONFIG.weight_decay,    learning_rate=CONFIG.learning_rate,
        max_epochs=CONFIG.max_epochs,        activation=CONFIG.activation,  
        initializer=CONFIG.initializer,      loss_function=CONFIG.loss_function,
        num_val=validation_size
    )

        training_loss, training_accuracy, validation_accuracy, predictions = NN.optimizer(
        NN.max_epochs, NN.num_train, NN.batch_size, NN.learning_rate)
    
wandb.agent(sweep_id, train, count= 1)




