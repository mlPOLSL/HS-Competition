import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

from jsonFUNFUNFUN.createFeatures import create_features
from loadJson import load_data
from paths import trainingSetJSON100k,testSetJSON100K,testSetLabelsJSON100K, deprecated_test_labels_path, original_deprecated_testpaths


X = create_features(load_data(trainingSetJSON100k),"list")
X_train = X[:, 0:35]
y_train = X[:, 35]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = Normalizer(copy=False)
clf = MLPClassifier(hidden_layer_sizes=(7,3), verbose=True)
pipe = Pipeline([('scaler', scaler), ('clf', clf)])
pipe.fit(X_train, y_train)

print(pipe.score(X_test, y_test))

