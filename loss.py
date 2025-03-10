import numpy as np



# Loss functions
def meanSquaredErrorLoss(self, Y_true, Y_pred):
        MSE = np.mean((Y_true - Y_pred) ** 2)
        return MSE

def crossEntropyLoss(self, Y_true, Y_pred):
        CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
        crossEntropy = np.mean(CE)
        return crossEntropy

def L2RegularisationLoss(self, weight_decay):
        ALPHA = weight_decay
        return ALPHA * np.sum(
            [
                np.linalg.norm(self.weights[str(i + 1)]) ** 2
                for i in range(len(self.weights))
            ]
        )






