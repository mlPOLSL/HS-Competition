import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from paths import testSet_paths,trainingSet_paths,deprecated_test_labels_path, testSet_paths_v2, testSet_paths_v3

def collide_testset(paths):
    X = np.loadtxt(paths[0], delimiter=',')
    X1 = np.loadtxt(paths[1], delimiter=',')
    X = np.vstack([X, X1])
    X2 = np.loadtxt(paths[2], delimiter=',')
    X = np.vstack([X, X2])
    return X

X = np.loadtxt("trainingSet_100k_v3.gz", delimiter=',')
X_train = X[:, 0:33]
y_train = X[:, 33]
y_test = np.loadtxt(deprecated_test_labels_path)
scaler = Normalizer(copy=False)
clf = MLPClassifier(hidden_layer_sizes=(7,3), verbose=True)
pipe = Pipeline([('scaler', scaler), ('clf', clf)])
pipe.fit(X_train, y_train)
X_test = collide_testset(testSet_paths_v3)
print(pipe.score(X_test, y_test))
# proba = pipe.predict_proba(X_test)

# with open("results_myller_v2.txt", 'w') as file:
#     for prob in proba:
#         file.write(str(prob[0]) + '\n')


