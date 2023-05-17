import numpy as np
from .train import grf

def calcW (h, l , p, y):
    m1 = np.solve(np.matmul(h.transpose(), l*np.diag(p)))

    return np.matmul(np.matmul(m1, h.transpose()), y)

def RBF(xin, m, covlist, w):

    N, n = xin.shape
    p = len(covlist)

    H = np.zeros((N, p))
    for j in range(N):
        for i in range(p):
            mi = m[i]
            covi = covlist[i]
            covi = np.array(covi).reshape(n,n) + 0.001*np.eye(n)
            H[j, i] = grf(xin[j,:], mi, covi, n)

    Haug = np.hstack((np.ones((N,1)), H))

    Yhat = np.dot(Haug, w)

    return Yhat


