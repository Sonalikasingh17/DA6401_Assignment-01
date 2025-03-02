import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}
        self.initialize_parameters()
    
    def initialize_parameters(self):
        for i in range(1, len(self.layer_sizes)):
            self.weights[i] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * 0.01
            self.biases[i] = np.zeros((self.layer_sizes[i], 1))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward_propagation(self, X):
        activations = {0: X}
        pre_activations = {}
        for i in range(1, len(self.layer_sizes)):
            z = self.weights[i] @ activations[i-1] + self.biases[i]
            pre_activations[i] = z
            activations[i] = self.softmax(z) if i == len(self.layer_sizes) - 1 else self.relu(z)
        return activations, pre_activations
    
    def compute_loss(self, Y_hat, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    
    def backward_propagation(self, activations, pre_activations, Y):
        gradients = {}
        m = Y.shape[1]
        dz = activations[len(self.layer_sizes) - 1] - Y
        for i in reversed(range(1, len(self.layer_sizes))):
            gradients[f'dW{i}'] = dz @ activations[i-1].T / m
            gradients[f'db{i}'] = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                dz = (self.weights[i].T @ dz) * self.relu_derivative(pre_activations[i-1])
        return gradients
    
    def update_parameters(self, gradients):
        for i in range(1, len(self.layer_sizes)):
            self.weights[i] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[i] -= self.learning_rate * gradients[f'db{i}']
    
    def train(self, X, Y, epochs=100, batch_size=64):
        losses = []
        for epoch in range(epochs):
            activations, pre_activations = self.forward_propagation(X)
            loss = self.compute_loss(activations[len(self.layer_sizes)-1], Y)
            losses.append(loss)
            gradients = self.backward_propagation(activations, pre_activations, Y)
            self.update_parameters(gradients)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
        return losses
    
    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[len(self.layer_sizes)-1], axis=0)

# Load Data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train.reshape(X_train.shape[0], -1) / 255.0, X_test.reshape(X_test.shape[0], -1) / 255.0
y_train, y_test = pd.get_dummies(y_train).values.T, pd.get_dummies(y_test).values.T

# Define model
nn = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
losses = nn.train(X_train.T, y_train, epochs=100)

# Plot loss
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

# Evaluate
predictions = nn.predict(X_test.T)
accuracy = np.mean(predictions == np.argmax(y_test, axis=0))
print(f'Test Accuracy: {accuracy * 100:.2f}%')



import numpy as np
import pandas as pd
import wandb
from tensorflow.keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="fashion-mnist-nn", name="simple-nn")

class SimpleNeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}
        self.init_params()
    
    def init_params(self):
        for i in range(1, len(self.layers)):
            self.weights[i] = np.random.randn(self.layers[i], self.layers[i-1]) * 0.01
            self.biases[i] = np.zeros((self.layers[i], 1))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        activations = {0: X}
        pre_activations = {}
        for i in range(1, len(self.layers)):
            z = self.weights[i] @ activations[i-1] + self.biases[i]
            pre_activations[i] = z
            activations[i] = self.softmax(z) if i == len(self.layers) - 1 else self.relu(z)
        return activations, pre_activations
    
    def loss(self, Y_hat, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m
    
    def backward(self, activations, pre_activations, Y):
        grads = {}
        m = Y.shape[1]
        dz = activations[len(self.layers) - 1] - Y
        for i in reversed(range(1, len(self.layers))):
            grads[f'dW{i}'] = dz @ activations[i-1].T / m
            grads[f'db{i}'] = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                dz = (self.weights[i].T @ dz) * self.relu_derivative(pre_activations[i-1])
        return grads
    
    def update(self, grads):
        for i in range(1, len(self.layers)):
            self.weights[i] -= self.learning_rate * grads[f'dW{i}']
            self.biases[i] -= self.learning_rate * grads[f'db{i}']
    
    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            activations, pre_activations = self.forward(X)
            loss = self.loss(activations[len(self.layers)-1], Y)
            grads = self.backward(activations, pre_activations, Y)
            self.update(grads)
            wandb.log({"epoch": epoch, "loss": loss})
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[len(self.layers)-1], axis=0)

# Load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
y_train = pd.get_dummies(y_train.flatten()).values.T
y_test = pd.get_dummies(y_test.flatten()).values.T

# Define model
nn = SimpleNeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
nn.train(X_train.T, y_train, epochs=100)

# Evaluate
predictions = nn.predict(X_test.T)
accuracy = np.mean(predictions == np.argmax(y_test, axis=0))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
wandb.log({"Test Accuracy": accuracy * 100})
