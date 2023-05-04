# Exercício 2 - Q3
# Daniel Rodrigues Coutinho - 2018020484

# Imports
from matplotlib import pyplot as plt
import numpy as np

# Mapeia os valores de x para as potencias
def getHMatrix(xs, degree):
    return np.matrix([[float(x**p) for p in range(degree, -1, -1)] for x in xs])

# Treina o modelo e fornece a matriz de pesos
def train(x, y, degree):
    h = getHMatrix(x, degree)
    pseudoInverse = np.linalg.pinv(h)
    w = np.matmul(pseudoInverse, np.matrix(y).transpose())
    return w

# Aplica o modelo as novas entradas
def applyModel(x, w):
    h = getHMatrix(x, degree)
    return np.matmul(h, w)


if __name__ == "__main__":

    # Treina o modelo

    samples = 100
    noise = np.random.normal(0, 4, samples)

    # Gera os pontos
    genFunc = lambda x: (x**2) * 0.5 + x * 3 + 10

    # Gera os dados de treinamento
    xt = np.linspace(-15, 10, num=samples)
    yt = genFunc(xt) + noise

    for degree in range(1,9):

        fig, sp = plt.subplots()

        # Obtem os pesos de treinamento
        w = train(xt, yt, degree)

        # Plota o polinomio gerado
        xp = np.linspace(-15, 10, num=1000)
        yp = applyModel(xp, w)

        # Plota os dados de treinamento
        sp.plot(xt, yt, "or", label="Samples")

        # Plota os dados obtidos
        sp.plot(xp, yp, "-b", label="Y_hat")

        # Plota a funcao geradora
        xg = np.linspace(-15, 10, num=1000)
        sp.plot(xg, list(map(genFunc, xg)), ":m", label="Target function")


    # Mostra os graficos obtidos
    plt.show()
