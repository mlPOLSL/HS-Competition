from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
trainingSet_paths = ["C:\\Users\\user\PycharmProjects\Hearthstone\\trainingSet1.gz",
                    "C:\\Users\\user\PycharmProjects\Hearthstone\\trainingSet2.gz",
                    "C:\\Users\\user\PycharmProjects\Hearthstone\\trainingSet3.gz",
                    "C:\\Users\\user\PycharmProjects\Hearthstone\\trainingSet4.gz"]
testSet_paths = ["C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testSet1.gz",
                 "C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testSet2.gz",
                 "C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testSet3.gz"]

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

X = collide_trainingset(trainingSet_paths)
X_train = X[100000:,0:18]
y_train = X[100000:,18]
forest = RandomForestClassifier(n_estimators=10, random_state=0)
net = MLPClassifier(hidden_layer_sizes=(7,3))
forest.fit(X_train, y_train)
model = SelectFromModel(estimator=forest, threshold=0.1, prefit=True)
X_selected = model.transform(X_train)
net.fit(X_selected, y_train)
X_test = collide_testset(testSet_paths)
y_test = np.loadtxt("C:\\Users\\user\PycharmProjects\Hearthstone\deprecated_testLabels\deprecated_testLabels.txt")
X_test = model.transform(X_test)
print("Accuracy: " + str(net.score(X_test, y_test)))
# importances = forest.feature_importances_
# for imp in importances:
#     print(imp)