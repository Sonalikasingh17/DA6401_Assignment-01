import numpy as np

def cross_entropy(label, y_pred):
    if y_pred[label] < 10 ** -8:
        return -np.log(10 ** -8)
    return -np.log(y_pred[label])


def squared_error(label, y_pred):
    y_true = np.zeros_like(y_pred)
    y_true[label] = 1
    size = float(len(y_pred))
    return np.array([(np.linalg.norm(y_true - y_pred) ** 2) / size]).reshape((1, 1))



