import numpy as np
import wandb
import time

# Optimizers
def sgd(self, epochs, length_dataset, learning_rate, weight_decay=0):

    # Implements Stochastic Gradient Descent (SGD) for training the neural network.
    
    # Arguments:
    #     : epochs - Number of training epochs
    #     : length_dataset - Number of training samples to use
    #     : learning_rate - Step size for weight updates
    #     : weight_decay - Regularization parameter (default: 0)
    
    # Returns:
    #     : training_loss - List of loss values over epochs
    #     : training_accuracy - List of training accuracies
    #     : validation_accuracy - List of validation accuracies
    #     : final_predictions - Model predictions after training
    

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
                epoch,
                training_loss[epoch],
                training_accuracy[epoch],
                validation_accuracy[epoch],
                elapsed_time,
                learning_rate,
            )
        )

            wandb.log({'loss': np.mean(batch_loss),
                   'training_accuracy': training_accuracy[epoch],
                   'validation_accuracy': validation_accuracy[epoch],
                   'epoch': epoch})

        return training_loss, training_accuracy, validation_accuracy, predictions

   
def mgd(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
    
    # Implements Mini-Batch Gradient Descent (MGD) for training the neural network.

    # Arguments:
    #     epochs : Number of training epochs.
    #     length_dataset : Number of training samples to use.
    #     batch_size : Mini-batch size for gradient updates.
    #     learning_rate : Step size for weight updates.
    #     weight_decay : Regularization parameter (default: 0).

    # Returns:
    #     training_loss, training_accuracy, validation_accuracy, final_predictions
      
    
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
                epoch,
                training_loss[epoch],
                training_accuracy[epoch],
                validation_accuracy[epoch],
                elapsed_time,
                learning_rate,
            )
        )

            wandb.log({
            'loss': np.mean(batch_loss),
            'training_accuracy': training_accuracy[epoch],
            'validation_accuracy': validation_accuracy[epoch],
            'epoch': epoch
        })

        return training_loss, training_accuracy, validation_accuracy, predictions
    
def ngd(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
    
    # Implements Nesterov Accelerated Gradient (NGD) for training the neural network.

    # Arguments:
    #     epochs : Number of training epochs.
    #     length_dataset: Number of training samples to use.
    #     batch_size : Mini-batch size for gradient updates.
    #     learning_rate: Step size for weight updates.
    #     weight_decay : Regularization parameter (default: 0).

    # Returns:
    #     training_loss, training_accuracy, validation_accuracy, final_predictions
    
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
                epoch,
                training_loss[epoch],
                training_accuracy[epoch],
                validation_accuracy[epoch],
                elapsed_time,
                learning_rate
            )
        )
        
            wandb.log({
            'loss': np.mean(batch_loss),
            'training_accuracy': training_accuracy[epoch],
            'validation_accuracy': validation_accuracy[epoch],
            'epoch': epoch
        })
        
        return training_loss, training_accuracy, validation_accuracy, predictions

    
def rmsProp(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0): 
    
    # Implements RMSProp optimizer for training the neural network.

    # Arguments:
    #     epochs : Number of training epochs.
    #     length_dataset : Number of training samples to use.
    #     batch_size: Mini-batch size for gradient updates.
    #     learning_rate : Step size for weight updates.
    #     weight_decay: Regularization parameter (default: 0).

    # Returns:
    #     training_loss, training_accuracy, validation_accuracy, final_predictions
    

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
                epoch,
                training_loss[epoch],
                training_accuracy[epoch],
                validation_accuracy[epoch],
                elapsed_time,
                learning_rate
            )
                )

            wandb.log({
            'loss': np.mean(batch_loss),
            'training_accuracy': training_accuracy[epoch],
            'validation_accuracy': validation_accuracy[epoch],
            'epoch': epoch
               })

        return training_loss, training_accuracy, validation_accuracy, predictions
    
def adam(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        
        # Implements the Adam optimizer for training the neural network.

        # Arguments:
        #     epochs : Number of training epochs.
        #     length_dataset : Number of training samples to use.
        #     batch_size: Mini-batch size for gradient updates.
        #     learning_rate : Step size for weight updates.
        #     weight_decay : Regularization parameter (default: 0).

        # Returns:
        #     training_loss, training_accuracy, validation_accuracy, final_predictions
    
        
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
                    epoch,
                    training_loss[epoch],
                    training_accuracy[epoch],
                    validation_accuracy[epoch],
                    elapsed_time,
                    learning_rate,
                )
            )

            wandb.log({
                'loss': np.mean(batch_loss),
                'training_accuracy': training_accuracy[epoch],
                'validation_accuracy': validation_accuracy[epoch],
                'epoch': epoch
            })

        return training_loss, training_accuracy, validation_accuracy, predictions
