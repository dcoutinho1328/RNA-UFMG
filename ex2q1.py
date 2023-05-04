# Exercício 2 - Q1
# Daniel Rodrigues Coutinho - 2018020484

import numpy as np
import matplotlib.pyplot as plt

def powerTwo(x):
    return x**2

def applyWeight(x, y, w):
    return w[1]*x + w[2]*y + w[0]

samples = 20

p = np.linspace(-1, 1, samples)

x, y = p, p

# Linearização

hx = powerTwo(x)
hy = powerTwo(y)

# Supondo a estimação de um neurônio
# No caso do problema

w = [-0.36, 1, 1]

# Plot all points
X, Y = np.meshgrid(hx, hy)

# Classifica os pontos utilizando o vetor de pesos
for i in range(samples):
    for j in range(samples):
        color = 'or' if applyWeight(X[i, j], Y[i, j], w) > 0 else 'ob'
        plt.plot(x[i], y[j], color)

# Plota a superfície de separação

p =np.linspace(-1, 1, 1000)
p_plus = np.linspace(-1, 0, 500)
p_minus = np.linspace(0, 1, 500)

xs = []
ys = []

# Encontra os pontos do circulo
for i in p:
    for j in p_minus:
        if abs(applyWeight(powerTwo(i), powerTwo(j), w)) <= 0.01:
            xs.append(i)
            ys.append(j)

for i in np.flip(p):
    for j in p_plus:
        if abs(applyWeight(powerTwo(i), powerTwo(j), w)) <= 0.01:
            xs.append(i)
            ys.append(j)

# Plota o circulo
plt.plot(xs, ys, '-g')

plt.show()
