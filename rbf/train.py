import numpy as np
from sklearn.cluster import KMeans

def grf(x, m, k, n):

    if (n == 1):
        r = np.sqrt(k)
        px = (1/(np.sqrt(2*np.pi*r*r))) * np.exp(-0.5 * np.power((x-m)/r, 2))
    else:
        px = (1/(np.sqrt((2*np.pi)**n * (np.linalg.det(k))))) * np.exp(-0.5 * np.dot(np.dot((x-m).transpose(), np.linalg.inv(k)), x-m))
    
    return px

def trainRBF(xin, yin, p):

    N, n = xin.shape

    xclust = KMeans(n_clusters=p).fit(np.asarray(xin))

    m = xclust.cluster_centers_
    covlist = []

    for i in range(p):
        ici = np.where(xclust.labels_ == i)
        xci = xin[ici,].reshape(-1, 2)
        # print(xin)
        # print('---------------')
        # print(ici)
        # print('---------------')
        # print(xci)
        if n == 1:
            covi = np.var(xci)
        else:
            covi = np.cov(xci).tolist()
        covlist.append(covi)

    H = []

    for i in range(N):
        H.append([])
        for j in range(p):
            mi = m[j,]
            covi = covlist[j]
            covi = np.reshape(np.ravel(np.matrix(covlist[j])), ) + 0.001 * np.eye(n)
            H[i].append(grf(xin[i,], mi, covi, n))

    H = np.matrix(H)

    extra = np.matrix([1 for _ in range(H.shape[0])]).transpose()
    H = np.hstack((extra, xin))

    W = np.dot(np.dot(np.linalg.inv(np.dot(H.transpose(), H)), H.transpose), yin)

    return W, H, m, covlist



        



