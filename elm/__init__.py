import numpy as np

# def elmToPoint(x, Z, W, par):


def elm(xin, Z, W, par=0):

    m, _ = xin.shape

    if par:
        extra = np.matrix([1 for _ in range(m)]).transpose()
        x_in = np.hstack((extra, xin))

    H = np.tanh(np.matmul(x_in, Z))

    f = np.vectorize(lambda x: np.sign(x) or 1)

    Y_hat = f(np.matmul(H, W))

    return Y_hat