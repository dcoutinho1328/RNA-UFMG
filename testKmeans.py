from sklearn.cluster import KMeans
import numpy as np

# Criação do conjunto de dados de exemplo
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Criação do objeto KMeans
kmeans = KMeans(n_clusters=2, random_state=0)

# Execução do algoritmo k-means
kmeans.fit(X)

# Atribuição dos clusters para cada observação
labels = kmeans.labels_

# Coordenadas dos centróides finais
centroids = kmeans.cluster_centers_

# Erros quadrados médios para cada grupo
inertia = kmeans.inertia_

print(labels)
print(centroids)
print(inertia)