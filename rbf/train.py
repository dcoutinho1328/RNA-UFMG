import numpy as np
from sklearn.cluster import KMeans

def grf(x, m, k, n):

    if (n == 1):
        r = np.sqrt(k)
        px = (1/(np.sqrt(2*np.pi*r*r))) * np.exp(-0.5 * np.power((x-m)/r, 2))
    else:
        px = (1/(np.sqrt((2*np.pi)**n * (np.linalg.det(k))))) * np.exp(-0.5 * np.dot(np.dot((x-m), np.linalg.inv(k)), (x-m).T))
    
    return px

def trainRBF(xin, yin, p):

    N, n = xin.shape

    xclust = KMeans(n_clusters=p).fit(np.asarray(xin))

    m = xclust.cluster_centers_
    covlist = []

    for i in range(p):
        ici = np.where(xclust.labels_ == i)[0]
        xci = xin[ici,:]
        covi = np.cov(xci.T)
        covlist.append(covi)

    H = np.zeros((N, p))
    for j in range(N):
        for i in range(p):
            mi = m[i]
            covi = covlist[i]
            covi = np.array(covi).reshape(n,n) + 0.001*np.eye(n)
            H[j, i] = grf(xin[j,:], mi, covi, n)

    Haug = np.hstack((np.ones((N,1)), H))

    W = np.dot(np.dot(np.linalg.inv(np.dot(Haug.T, Haug)), Haug.T), yin)

    return W, H, m, covlist



        



