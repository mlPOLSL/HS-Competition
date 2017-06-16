import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from paths import testSet_paths,trainingSet_paths_v4,deprecated_test_labels_path

NO_OF_FEATURES = 36

def load_batch(path):
    dataset = np.loadtxt(path, delimiter=',')
    return dataset

def get_minibatch(dataset, range, no_of_features):
    try:
        X, y = dataset[range[0]:range[1], 0:no_of_features - 1], dataset[range[0]:range[1], no_of_features - 1]
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


def partial_learn(classifier, scaler, initial_range, classes, trainingset_paths, no_of_features):
    for path in trainingset_paths:
        batch = load_batch(path)
        print("another batch")
        for i in range(0, 50):
            print("another minibatch")
            minibatch_range = [initial_range * i, initial_range * (i + 1)]
            X_train, y_train = get_minibatch(batch, minibatch_range, no_of_features)
            if X_train.size == 0:
                break
            X_train = scaler.fit_transform(X_train)
            classifier.partial_fit(X_train, y_train, classes)
    X_test = collide_testset(testSet_paths)
    y_test = np.loadtxt(deprecated_test_labels_path)
    X_test = scaler.transform(X_test)
    print("Accuracy: " + str(classifier.score(X_test, y_test)))
    return classifier.predict_proba(X_test)


clf = MLPClassifier()
scaler = StandardScaler()
probas = partial_learn(clf, scaler, 10000, np.array([0, 1]), trainingSet_paths_v4, NO_OF_FEATURES)
