import numpy as np

def bestStump(x, y, weights):
    n_features = x.shape[1]
    best_stump, best_threshold, best_error = None, None, float('inf')
    for i in range(n_features):
        min_val, max_val = x[:, i].min(), x[:, i].max()
        thresholds = np.linspace(min_val, max_val, 3)
        for t in thresholds:
            pred = np.where((x[:, i] < t) | (x[:, i] > t), 1, -1)
            error = np.sum(weights[pred != y])
            if error < best_error:
                best_threshold, best_error, best_stump = t, error, i
    return best_stump, best_threshold, best_error

def adaboost_predict(X, classifiers, alphas):
    final_pred = np.zeros(X.shape[0])
    for alpha, (feature, threshold) in zip(alphas, classifiers):
        predictions = np.where(X[:, feature] <= threshold, 1, -1)
        final_pred += alpha * predictions
    return np.sign(final_pred)

def fit_stump(x, grad, num_splits=20):
    thresholds = np.linspace(0, 1, num_splits)
    best_threshold, best_error, best_left, best_right = None, float('inf'), None, None
    for t in thresholds:
        left = grad[x <= t].mean() if np.any(x <= t) else 0
        right = grad[x > t].mean() if np.any(x > t) else 0
        pred = np.where(x <= t, left, right)
        error = np.sum((grad - pred) ** 2)
        if error < best_error:
            best_error, best_threshold, best_left, best_right = error, t, left, right
    return best_threshold, best_left, best_right

def gradient_boosting(x_train, y_train, x_test, y_test, loss, lr=0.01, iterations=100):
    y_pred_train, y_pred_test = np.zeros_like(y_train), np.zeros_like(y_test)
    train_loss, test_loss = [], []
    for i in range(iterations):
        grad = (y_train - y_pred_train) if loss == "squared" else np.sign(y_train - y_pred_train)
        threshold, left, right = fit_stump(x_train, grad)
        stump_pred_train = np.where(x_train <= threshold, left, right)
        stump_pred_test = np.where(x_test <= threshold, left, right)
        y_pred_train += lr * stump_pred_train
        y_pred_test += lr * stump_pred_test
        if loss == "squared":
            train_loss.append(np.mean((y_train - y_pred_train) ** 2))
            test_loss.append(np.mean((y_test - y_pred_test) ** 2))
        else:
            train_loss.append(np.mean(np.abs(y_train - y_pred_train)))
            test_loss.append(np.mean(np.abs(y_test - y_pred_test)))
    return y_pred_train, y_pred_test, train_loss, test_loss
