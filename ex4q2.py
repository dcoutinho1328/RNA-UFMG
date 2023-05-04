import numpy as np
import matplotlib.pyplot as plt
from perceptron.train import trainPerceptron
from perceptron import perceptron
from tabulate import tabulate

# Define o desvio padrão
desvio = 0.3

# Define o número de pontos a serem gerados
num_pontos = 200

# Gera uma distribuição normal de pontos ao redor do ponto central
g1 = np.hstack((np.random.normal(loc=[2,2], scale=desvio, size=(num_pontos, 2)), np.matrix([0 for _ in range(num_pontos)]).transpose()))
g2 = np.hstack((np.random.normal(loc=[4,4], scale=desvio, size=(num_pontos, 2)), np.matrix([1 for _ in range(num_pontos)]).transpose()))

data = np.vstack((g1, g2))
np.random.shuffle(data)

r, c = data.shape

# Calcula o numero de amostras de treinamento
tsn = int(np.round(0.7 * r))

x_in = data[:tsn, :2]
y_in = data[:tsn, 2]

w, _ = trainPerceptron(x_in, y_in, 0.01, 0.01, 50, 1)

x_test = data[tsn:, :2]
y_target = data[tsn:, 2]

y_o_train = perceptron(x_in, w, 1)
y_o = perceptron(x_test, w, 1)

diff = y_target - y_o
diff_train = y_in - y_o_train

# Calcula a acuracia
ac = 1 - np.count_nonzero(diff)/(r-tsn)
ac_train = 1 - np.count_nonzero(diff_train)/tsn

# Calcula o numero de amostras
# classificadas corretamente da classe 2
nc2 = np.count_nonzero(y_target)
nc2_train = np.count_nonzero(y_in)

# Calcula o numero de amostras
# classificadas de forma incorreta
c1Asc2 = np.count_nonzero(diff == -1)
c2Asc1 = np.count_nonzero(diff == 1)
c1Asc2_train = np.count_nonzero(diff_train == -1)
c2Asc1_train = np.count_nonzero(diff_train == 1)

# Monta a matriz de confusão
conf = [["C1"],["C2"]]
conf_train = [["C1"],["C2"]]

conf_train[0].append(tsn - nc2_train - c1Asc2_train)
conf_train[0].append(c1Asc2_train)
conf_train[1].append(c2Asc1_train)
conf_train[1].append(nc2_train - c2Asc1_train)

conf[0].append(r - tsn - nc2 - c1Asc2)
conf[0].append(c1Asc2)
conf[1].append(c2Asc1)
conf[1].append(nc2 - c2Asc1)

with open("results_e4q2.txt", 'w') as f:
    f.write("Matriz de confusao de treinamento: \n \n")
    f.write(tabulate(conf_train, headers=['', 'C1', 'C2'], tablefmt='orgtbl'))
    f.write("\n \n")
    f.write(f"Acuracia de treinamento: {ac_train:.2%}\n \n")
    f.write('----------------------------------- \n \n')
    f.write("Matriz de confusao de testes: \n \n")
    f.write(tabulate(conf, headers=['', 'C1', 'C2'], tablefmt='orgtbl'))
    f.write("\n \n")
    f.write(f"Acuracia de testes: {ac:.2%}\n")