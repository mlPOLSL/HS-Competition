from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer


X = np.loadtxt("trainingSet_100k.gz", delimiter=',')
X_train = X[:, 0:25]
y_train = X[:, 25]

pca = PCA(n_components=2)
scaler = Normalizer()
X_train = scaler.fit_transform(X_train)
X_train = pca.fit_transform(X_train)
X_train = X_train[11500:13500,:]

for i, row in enumerate(X_train, 11500):
    if y_train[i] == 1:
        plt.scatter(row[0], row[1], c='r', marker='^')
    elif y_train[i] == 0:
        plt.scatter(row[0], row[1], c='b')

plt.show()

