import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1, random_state=42)

# Transformação para gerar as espirais
t = np.linspace(0, 1.5*np.pi, 1000)
spiral = np.column_stack((np.cos(t), np.sin(t))) * t.reshape((-1,1))
X = np.dot(X, spiral.T)

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.show()
