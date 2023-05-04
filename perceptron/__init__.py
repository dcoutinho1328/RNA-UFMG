import numpy as np

def perceptron(x, w, par):

    m, _ = x.shape

    if par:
        extra = np.matrix([-1 for _ in range(m)]).transpose()
        x = np.hstack((extra, x))

    f = np.vectorize(lambda t: int(t >= 0))

    return f(np.matmul(x, w))

def visualizeSurface3D(w):
    
    w0, w1, w2 = map(float, w.transpose().tolist()[0])

    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)

    X, Y = np.meshgrid(x, y)

    f = lambda x, y: w1*x + w2*y - w0 >= 0        

    Z = f(X, Y)

    return X, Y, Z

def visualizeSurface2D(w):
    
    w0, w1, w2 = map(float, w.transpose().tolist()[0])

    x = np.linspace(0, 6, 100)

    f = lambda x: -x*w1/w2 + w0/w2

    y = f(x)

    return x, y




# def adaline(x_in, w, par=0):

#     m, _ = shape(x_in)

#     if par:
#         extra = matrix([1 for _ in range(m)]).transpose()
#         x_in = hstack((x_in, extra))

#     y = matmul(x_in, w)

#     return y