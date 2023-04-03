import numpy as np
import matplotlib.pyplot as plt
from perceptron.train import trainPerceptron
from perceptron import visualizeSurface2D, visualizeSurface3D

# Define o desvio padrão
desvio = 0.3

# Define o número de pontos a serem gerados
num_pontos = 20

# Gera uma distribuição normal de pontos ao redor do ponto central
g1 = np.hstack((np.random.normal(loc=[2,2], scale=desvio, size=(num_pontos, 2)), np.matrix([0 for _ in range(num_pontos)]).transpose()))
g2 = np.hstack((np.random.normal(loc=[4,4], scale=desvio, size=(num_pontos, 2)), np.matrix([1 for _ in range(num_pontos)]).transpose()))

data = np.vstack((g1, g2))
np.random.shuffle(data)

x_in = data[:, :2]
y_in = data[:, 2]

w, _ = trainPerceptron(x_in, y_in, 0.01, 0.01, 50, 1)
x, y = visualizeSurface2D(w)

plt.figure()
plt.plot(g1[:, 0], g1[:, 1], 'or')
plt.plot(g2[:, 0], g2[:, 1], 'ob')
plt.plot(x, y, '-m')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(g1[:, 0], g1[:, 1], g1[:, 2], color='red')
ax.scatter(g2[:, 0], g2[:, 1], g1[:, 2], color="blue")

x, y, z = visualizeSurface3D(w)

ax.plot_surface(x, y, z)

plt.show()

