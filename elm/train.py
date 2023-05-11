import numpy as np

def calcW (h, l, p, y):
    m1 = np.linalg.inv(np.dot(h.transpose(), h) + l*np.eye(p))
    # m2 = np.dot(np.dot(np.linalg.inv((np.dot(np.transpose(h), h) + l*np.identity(p))),np.transpose(h)), y)
    return np.dot(np.dot(m1, h.transpose()), y)
    # return m2

def trainElm2(xin, yin, p, l, par=0):
    
    Z = []

    for _ in range(p):
        Z.append(np.random.uniform(-0.5, 0.5, size = 3))

    Z = np.matrix(Z).transpose()

    m, _ = xin.shape

    if par:
        extra = np.matrix([1 for _ in range(m)]).transpose()
        x_in = np.hstack((extra, xin))

    H = np.tanh(np.matmul(x_in, Z))

    W = calcW(H, l, p, yin)

    return W, H, Z

def trainElm(xin, yin, p, par = 0):

    Z = []

    for _ in range(p):
        Z.append(np.random.uniform(-0.5, 0.5, size = 3))

    Z = np.matrix(Z).transpose()

    m, _ = xin.shape

    if par:
        extra = np.matrix([1 for _ in range(m)]).transpose()
        x_in = np.hstack((extra, xin))

    H = np.tanh(np.matmul(x_in, Z))

    W = np.matmul(np.linalg.pinv(H), yin)

    return W, H, Z
