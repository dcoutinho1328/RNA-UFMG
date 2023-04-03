from numpy import matrix, random, matmul, hstack, shape
from random import shuffle

def trainPerceptron(xin, yin, eta, tol, maxEpoq, par=1):

    m, n = shape(xin)

    wt = []
    wt.append(random.uniform(size = n + par) - 0.5)
    wt = matrix(wt).transpose()

    if par:
        extra = matrix([-1 for _ in range(m)]).transpose()
        xin = hstack((extra, xin))


    nEpoq = 0
    errEpoq = tol + 1

    errorVector = [0 for _ in range(maxEpoq)]
    while (nEpoq < maxEpoq and errEpoq > tol):

        ei2 = 0
        xseq = list(range(m))
        shuffle(xseq)

        for i in range(m):
            irand = xseq[i]
            yhati = float(matmul(xin[irand], wt) >= 0)
            ei = float(yin[irand]) - yhati
            dw = matrix(xin[irand]*eta*ei).transpose()
            wt = wt + dw
            ei2 = ei2 + ei**2
        
        errorVector[nEpoq] = ei2/m
        nEpoq += 1

        errEpoq = float(errorVector[nEpoq - 1])
        
    return wt, errorVector[0:nEpoq]
        





