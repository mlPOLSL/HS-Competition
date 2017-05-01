import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from jsonFUNFUNFUN.createFeatures import create_features
from loadJson import load_data
from paths import trainingSetJSON100k,testSetJSON100K,testSetLabelsJSON100K

# numberOfPackages = "ALL"
numberOfPackages = 1
X = create_features(load_data(trainingSetJSON100k,numberOfPackages),"list")
X_train = X[:, 0:35]
y_train = X[:, 35]

X_test = create_features(load_data(testSetJSON100K,numberOfPackages),"list",'test')
y_test = np.loadtxt(testSetLabelsJSON100K)

scaler = Normalizer(copy=False)
clf = MLPClassifier(hidden_layer_sizes=(7,3), verbose=True)
pipe = Pipeline([('scaler', scaler), ('clf', clf)])
pipe.fit(X_train, y_train)

print(pipe.score(X_test, y_test))

