import numpy as np

def train_model(x_features, y_values):
    return np.linalg.inv(x_features.T @ x_features) @ x_features.T @ y_values

def predict(x_features, params):
    return x_features @ params
