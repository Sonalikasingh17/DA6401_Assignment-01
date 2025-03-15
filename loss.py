import numpy as np



# Loss functions

def mean_squared_error(self, true_labels, predicted_labels):
        return np.mean((true_labels - predicted_labels) ** 2)

def cross_entropy_loss(self, true_labels, predicted_labels):
        return np.mean([-true_labels[i] * np.log(predicted_labels[i]) for i in range(len(predicted_labels))])

def l2_regularisation_loss(self, weight_decay):
        return weight_decay * np.sum([np.linalg.norm(self.weights[str(i + 1)]) ** 2 for i in range(len(self.weights))])






