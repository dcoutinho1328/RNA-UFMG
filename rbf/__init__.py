import numpy as np
from .train import grf

def calcW (h, l , p, y):
    m1 = np.solve(np.matmul(h.transpose(), l*np.diag(p)))

    return np.matmul(np.matmul(m1, h.transpose()), y)

def RBF(xin, m, covlist, w):

    N, n = xin.shape
    p = len(covlist)

    H = []

    for i in range(N):
        H.append([])
        for j in range(p):
            mi = m[j,]
            covi = covlist[j]
            covi = np.matrix(np.array(covlist[i]).reshape(-1, n), copy=False) + 0.001 * np.eye(n)
            H[i].append(grf(xin[i,], mi, covi, n))

    H = np.matrix(H)

    extra = np.matrix([1 for _ in range(H.shape[0])]).transpose()
    H = np.hstack((extra, xin))

    Yhat = np.dot(H, w)

    return Yhat


