import numpy as np
from util import readFromFile
from os.path import dirname
from adaline.train import trainAdaline
from adaline import adaline
import matplotlib.pyplot as plt

# Define o diretório com os arquivos
dataDir = dirname(__file__) + "/data/ex3"

# Define os arquivos a serem lidos
x_file = dataDir + "/Ex1_x"
y_file = dataDir + "/Ex1_y"
t_file = dataDir + "/Ex1_t"

# Extrai os dados dos arquivos
x_f = readFromFile(x_file)
y_f = readFromFile(y_file)
t_f = readFromFile(t_file)

# Treina o modelo com os dados
w, _ = trainAdaline(x_f, y_f, 0.1, 0.01, 10000, 1)

# Aplica o modelo aos dados lidos
# Apenas para checar o resultado do treinamento
y_o = adaline(x_f, w, 1)

fig1, ax1 = plt.subplots()

# Plota os dados lidos
ax1.plot(t_f, y_f, '-b')
ax1.plot(t_f, x_f, '-g')

# Plota a saida do modelo aplicado aos
# dados de treinamento
ax1.plot(t_f, y_o, '-r')

fig2, ax2 = plt.subplots()

# Gera novos valores de t
newT = np.linspace(0,6,10)

# Gera novos valores de x para uma nova senoidal
x_r = 2*np.sin(np.matrix(newT).transpose())

# Aplica o modelo aos novos valores
y_o2 = adaline(x_r, w, 1)

# Plota os novos dados
ax2.plot(newT, x_r, '-m')
ax2.plot(newT, y_o2, '-y')

plt.show()