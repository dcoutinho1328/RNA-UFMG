from numpy import matrix, matmul, hstack, shape

def adaline(x_in, w, par=0):

    m, _ = shape(x_in)

    if par:
        extra = matrix([1 for _ in range(m)]).transpose()
        x_in = hstack((extra, x_in))

    y = matmul(x_in, w)

    return y