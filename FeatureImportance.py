from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from paths import testSet_paths_v3,trainingSet_paths,deprecated_test_labels_path

NO_OF_FEATURES = 22

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
forest = RandomForestClassifier(max_features=13, min_samples_leaf=50, min_samples_split=30)
X = np.loadtxt("C:\\Users\\user\PycharmProjects\Hearthstone\TrainingSet_10Folds_Final\\trainingSet3_Final.gz", delimiter=',')
X_train = X[:, 0:NO_OF_FEATURES]
y_train = X[:, NO_OF_FEATURES]
forest.fit(X_train, y_train)
importances = forest.feature_importances_
for imp in importances:
    print(imp)