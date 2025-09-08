import numpy as np

def fda(x, y):
    c = np.unique(y)
    u = np.mean(x, axis=0)
    sb, sw = np.zeros((x.shape[1], x.shape[1])), np.eye(x.shape[1]) * 1e-6
    for d in c:
        xc = x[y == d]
        uc = np.mean(xc, axis=0)
        sb += len(xc) * np.outer(uc - u, uc - u)
        sw += np.cov(xc, rowvar=False) * (len(xc) - 1)
    e, v = np.linalg.eigh(np.linalg.pinv(sw).dot(sb))
    w = v[:, np.argsort(e)[::-1][:2]]
    return np.dot(x, w), w

def lda_train(x, y):
    c = np.unique(y)
    u, s = {}, np.zeros((x.shape[1], x.shape[1]))
    for d in c:
        x_c = x[y == d]
        u[d] = np.mean(x_c, axis=0)
        s += np.cov(x_c, rowvar=False) * (len(x_c) - 1)
    s /= len(x) - len(c)
    return u, s

def lda_predict(x, u, s):
    inv_s = np.linalg.pinv(s)
    scores = {c: x @ inv_s @ u[c] - 0.5 * u[c].T @ inv_s @ u[c] for c in u}
    return np.array([max(scores, key=lambda c: scores[c][i]) for i in range(x.shape[0])])

def qda_train(x, y):
    c = np.unique(y)
    u, s = {}, {}
    for d in c:
        x_c = x[y == d]
        u[d] = np.mean(x_c, axis=0)
        s[d] = np.cov(x_c, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    return u, s

def qda_predict(x, u, s):
    scores = {}
    for c in u:
        inv_s = np.linalg.pinv(s[c])
        det_s = np.linalg.det(s[c])
        term1 = -0.5 * np.log(det_s)
        term2 = -0.5 * np.sum((x - u[c]) @ inv_s * (x - u[c]), axis=1)
        scores[c] = term1 + term2
    return np.array([max(scores, key=lambda c: scores[c][i]) for i in range(x.shape[0])])
