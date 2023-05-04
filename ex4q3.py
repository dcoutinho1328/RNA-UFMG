from sklearn import datasets
import numpy as np
from perceptron.train import trainPerceptron
from perceptron import perceptron
from tabulate import tabulate

# Carrega os dados do Iris
iris = datasets.load_iris()
X = iris.data
Y = np.where(iris.target == 2, 1, iris.target)

tx = X[:100]
ty = np.matrix(Y[:100]).transpose()

data = np.hstack((tx, ty))

def loop():
    np.random.shuffle(data)

    tx = data[:70, :-1]
    ty = data[:70, -1]

    w, _ = trainPerceptron(tx, ty, 0.01, 0.01, 50, 1)

    testx = data[70:, :-1]
    testy = data[70:, -1]

    y_o = perceptron(testx, w, 1)
    y_o_train = perceptron(tx, w, 1)

    diff = testy - y_o
    diff_train = ty - y_o_train

    # Calcula a acuracia
    ac = 1 - np.count_nonzero(diff)/30
    ac_train = 1 - np.count_nonzero(diff_train)/70

    # Calcula o numero de amostras
    # classificadas corretamente da classe 2
    nc2 = np.count_nonzero(testy)
    nc2_train = np.count_nonzero(ty)

    # Calcula o numero de amostras
    # classificadas de forma incorreta
    c1Asc2 = np.count_nonzero(diff == -1)
    c2Asc1 = np.count_nonzero(diff == 1)
    c1Asc2_train = np.count_nonzero(diff_train == -1)
    c2Asc1_train = np.count_nonzero(diff_train == 1)

    # Monta a matriz de confusão
    conf = [["C1"],["C2"]]
    conf_train = [["C1"],["C2"]]

    conf_train[0].append(70 - nc2_train - c1Asc2_train)
    conf_train[0].append(c1Asc2_train)
    conf_train[1].append(c2Asc1_train)
    conf_train[1].append(nc2_train - c2Asc1_train)

    conf[0].append(30 - nc2 - c1Asc2)
    conf[0].append(c1Asc2)
    conf[1].append(c2Asc1)
    conf[1].append(nc2 - c2Asc1)

    return ac, conf, ac_train, conf_train

if __name__ == "__main__":

    a, c, at, ct = loop()

    with open("results_e4q3.txt", 'w') as f:
        f.write("Matriz de confusao de treinamento: \n \n")
        f.write(tabulate(ct, headers=['', 'C1', 'C2'], tablefmt='orgtbl'))
        f.write("\n \n")
        f.write(f"Acuracia de treinamento: {at:.2%}\n \n")
        f.write('----------------------------------- \n \n')
        f.write("Matriz de confusao de testes: \n \n")
        f.write(tabulate(c, headers=['', 'C1', 'C2'], tablefmt='orgtbl'))
        f.write("\n \n")
        f.write(f"Acuracia de testes: {a:.2%}\n \n")

    acArray = []
    actArray = []

    for _ in range(100):

        a, _, at, _ = loop()
        acArray.append(a)
        actArray.append(at)

    # Calcula as medias
    avg = np.average(acArray)
    avgt = np.average(actArray)

    # Calcula as variancias
    v = np.average((np.array(acArray) - avg)**2)
    vt = np.average((np.array(actArray) - avgt)**2)

    with open("results_e4q3.txt", 'a') as f:
        f.write('----------------------------------- \n \n')
        f.write(f'------- {100} Treinamentos --------- \n \n')
        f.write(f"Media da acuracia de treinamento: {avg:.2%}\n \n")
        f.write(f"Variancia de treinamento: {v:.2%}\n \n")
        f.write('----------------------------------- \n \n')
        f.write(f"Media da acuracia de teste: {avgt:.2%}\n \n")
        f.write(f"Variancia de teste: {vt:.2%}\n \n")





