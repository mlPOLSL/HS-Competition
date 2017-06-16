import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib
from paths import trainingset_folds
from sklearn.base import clone


def pickler_classifiers(classifiers, trainingSet_folds, pickle_save_path):
    for i, fold in enumerate(trainingSet_folds, 1):
        X = np.loadtxt(fold, delimiter=',')
        X_train = X[:, 0:28]
        y_train = X[:, 28]
        for j, clf in enumerate(classifiers, 1):
            fitted_clf = clone(clf).fit(X_train, y_train)
            clf_name = type(fitted_clf).__name__
            path = pickle_save_path + str(clf_name) + '_' + str(
                j) + '_fold' + str(i) + '.pkl'
            joblib.dump(fitted_clf, path)


mlp1 = MLPClassifier(hidden_layer_sizes=(20,), learning_rate_init=0.001, alpha=0.1)
mlp2 = MLPClassifier(hidden_layer_sizes=(20, 10), learning_rate_init=0.001, alpha=0.1)
mlp3 = MLPClassifier(hidden_layer_sizes=(7, 3), learning_rate_init=0.01, alpha=0.001, tol=0.00000001)
LogReg1 = LogisticRegression(C=0.1, solver='lbfgs', tol=0.00000001, max_iter=1000)
LogReg2 = LogisticRegression(C=0.1, solver='newton-cg', tol=0.00000001, max_iter=1000)
LogReg3 = LogisticRegression(C=0.1, solver='sag', tol=0.00000001, max_iter=1000)
forest1 = RandomForestClassifier(max_features=13, min_samples_leaf=50, min_samples_split=30)
forest2 = RandomForestClassifier(max_features=14, min_samples_leaf=50, min_samples_split=30)
forest3 = RandomForestClassifier(min_samples_leaf=60, min_samples_split=80)

pipe1 = Pipeline([('scaler', StandardScaler()), ('clf', mlp1)])
pipe2 = Pipeline([('scaler', StandardScaler()), ('clf', mlp2)])
pipe3 = Pipeline([('scaler', StandardScaler()), ('clf', mlp3)])
pipe4 = Pipeline([('scaler', StandardScaler()), ('clf', LogReg1)])
pipe5 = Pipeline([('scaler', StandardScaler()), ('clf', LogReg2)])
pipe6 = Pipeline([('scaler', StandardScaler()), ('clf', LogReg3)])

all_clfs = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, forest1, forest2, forest3]

pickler_classifiers(all_clfs, trainingset_folds, "C:\\Users\\user\PycharmProjects\Hearthstone\PicklesFinal\\")
