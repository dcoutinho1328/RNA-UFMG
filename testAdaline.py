from adaline.train import trainAdaline
from adaline import adaline
import matplotlib.pyplot as plt
from numpy import linspace, matrix, hstack
from math import pi, sin, cos, tanh

samples = 20


# Unival

t = linspace(0, 2*pi, samples)

x_in = matrix([sin(x) for x in t]).transpose()

y_in = matrix(4*x_in + 2)

w, z = trainAdaline(x_in, y_in, 0.01, 0.01, 10000, 1)

y_o = adaline(x_in, w, 1)

fig, ax1 = plt.subplots()

ax1.plot(t, y_in, 'r-', linewidth=2.0)

ax1.plot(t, y_o, 'b-', linewidth=2.0)


# Multival

t = linspace(0, 2*pi, samples)

x1 = matrix([sin(x) + cos(x) for x in t]).transpose()
x2 = matrix([tanh(x) for x in t]).transpose()
x3 = matrix([sin(4*x) for x in t]).transpose()
x4 = matrix([abs(sin(x)) for x in t]).transpose()

x_in = hstack((x1,x2,x3,x4))

y_in = matrix(x1*3.2) + matrix(x2*0.8) + matrix(x3*2) + matrix(x4*(pi/2))

w, _ = trainAdaline(x_in, y_in, 0.01, 0.01, 50, 1)

y_o = adaline(x_in, w, 1)

fig2, ax2 = plt.subplots()

ax2.plot(t, y_in, 'r-', linewidth=2.0)

ax2.plot(t, y_o, 'b-', linewidth=2.0)

plt.show()

