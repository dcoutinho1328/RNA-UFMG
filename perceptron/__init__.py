from numpy import linspace, shape, meshgrid

def visualizeSurface3D(w):
    m, _ = shape(w)

    if m != 3:
        raise("Matriz de pesos não compatível")

    w0, w1, w2 = map(float, w.transpose().tolist()[0])

    x = linspace(0, 6, 100)
    y = linspace(0, 6, 100)

    X, Y = meshgrid(x, y)

    f = lambda x, y: w1*x + w2*y - w0 >= 0
        

    Z = f(X, Y)

    return X, Y, Z

def visualizeSurface2D(w):
    m, _ = shape(w)

    if m != 3:
        raise("Matriz de pesos não compatível")

    w0, w1, w2 = map(float, w.transpose().tolist()[0])

    x = linspace(0, 6, 100)

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