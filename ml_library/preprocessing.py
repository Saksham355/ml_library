import numpy as np

def build_features(x_values, degree):
    features = [x_values ** d for d in range(degree + 1)]
    return np.column_stack(features)

def pca(x, num_features=5):
    mean = np.mean(x, axis=0)
    x_centered = x - mean
    cov_matrix = np.dot(x_centered.T, x_centered) / x.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, indices[:num_features]]
    return np.dot(x_centered, top_eigenvectors)
