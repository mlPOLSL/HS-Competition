from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from paths import testSet_paths_v3,trainingSet_paths,deprecated_test_labels_path

def collide_trainingset(paths):
    X = np.loadtxt(paths[0], delimiter=',')
    # X1 = np.loadtxt(paths[1], delimiter=',')
    # X = np.vstack([X, X1])
    # X2 = np.loadtxt(paths[2], delimiter=',')
    # X = np.vstack([X, X2])
    # X3 = np.loadtxt(paths[3], delimiter=',')
    # X = np.vstack([X, X3])
    return X

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
forest = RandomForestClassifier(n_estimators=10, random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
for imp in importances:
    print(imp)