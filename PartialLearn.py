import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from paths import testSet_paths,trainingSet_paths,deprecated_test_labels_path

def load_batch(path):
    dataset = np.loadtxt(path, delimiter=',')
    return dataset

def get_minibatch(dataset, range):
    try:
        X, y = dataset[range[0]:range[1], 0:18], dataset[range[0]:range[1], 18]
    except ValueError:
        return None, None
    return X, y

def collide_testset(paths):
    X = np.loadtxt(paths[0], delimiter=',')
    X1 = np.loadtxt(paths[1], delimiter=',')
    X = np.vstack([X, X1])
    X2 = np.loadtxt(paths[2], delimiter=',')
    X = np.vstack([X, X2])
    return X

scaler = StandardScaler(copy=False)
clf = MLPClassifier(hidden_layer_sizes=(7, 3))
init_range = 10000
classes = np.array([0, 1])
for path in trainingSet_paths:
    batch = load_batch(path)
    print("another batch")
    for i in range(0, 50):
        print("another minibatch")
        minibatch_range = [init_range * i, init_range * (i + 1)]
        X_train, y_train = get_minibatch(batch, minibatch_range)
        if X_train.size == 0:
            break
        X_train = scaler.fit_transform(X_train)
        clf.partial_fit(X_train, y_train, classes)
X_test = collide_testset(testSet_paths)
y_test = np.loadtxt(deprecated_test_labels_path)
X_test = scaler.transform(X_test)
print("Accuracy: " + str(clf.score(X_test, y_test)))