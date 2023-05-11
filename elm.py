import numpy as np
import matplotlib.pyplot as plt
from elm.train import trainElm
from elm import elm

desvio = 0.6

num_pontos = 20

p = 5

g1_a = np.hstack((np.random.normal(loc=[2,2], scale=desvio, size=(num_pontos, 2)), np.matrix([-1 for _ in range(num_pontos)]).transpose()))
g1_b = np.hstack((np.random.normal(loc=[4,4], scale=desvio, size=(num_pontos, 2)), np.matrix([-1 for _ in range(num_pontos)]).transpose()))
g2_a = np.hstack((np.random.normal(loc=[2,4], scale=desvio, size=(num_pontos, 2)), np.matrix([1 for _ in range(num_pontos)]).transpose()))
g2_b = np.hstack((np.random.normal(loc=[4,2], scale=desvio, size=(num_pontos, 2)), np.matrix([1 for _ in range(num_pontos)]).transpose()))

g1 = np.vstack((g1_a, g1_b))
g2 = np.vstack((g2_a, g2_b))

plt.plot(g1[:, 0], g1[:, 1], 'or')
plt.plot(g2[:, 0], g2[:, 1], 'ob')

t = np.linspace(-2, 10, 120)

X = np.vstack((g1[:, :2], g2[:, :2]))
Y = np.vstack((g1[:, 2], g2[:, 2]))

W, H, Z = trainElm(X, Y, p, 1)

Xm, Ym = np.meshgrid(t, t)

ae = np.vectorize(lambda x, y: elm(np.matrix([x, y]), Z, W, 1))

Zm = ae(Xm, Ym)

print(Zm)

plt.contour(Xm, Ym, Zm)

plt.show()
