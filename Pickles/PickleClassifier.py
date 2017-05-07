import pickle
from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.loadtxt("trainingSet_100k_v4_v2.gz", delimiter=',')
X_train = X[:, 0:35]
y_train = X[:, 35]
clf = MLPClassifier(hidden_layer_sizes=(20,), learning_rate_init=0.001, alpha=0.1)
clf.fit(X_train, y_train)
