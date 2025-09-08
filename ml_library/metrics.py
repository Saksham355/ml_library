import numpy as np

def mean_squared_error(predicted, actual):
    return np.mean((predicted - actual) ** 2)

def compute_loss(y_true, y_pred):
    """Mean squared error (for neural networks)."""
    return np.mean((y_true - y_pred) ** 2)