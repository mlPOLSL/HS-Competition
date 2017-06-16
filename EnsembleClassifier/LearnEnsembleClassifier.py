from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from EnsembleClassifier.MajorityVoteClassifier import MajorityVoteClassifier
import numpy as np
from sklearn.externals import joblib
import glob
from paths import testSet_paths, pickle_classifiers

pickles_dir = "C:\\Users\\user\PycharmProjects\Hearthstone\PicklesFinal\\"

def collide_testset(paths):
    X = np.loadtxt(paths[0], delimiter=',')
    X1 = np.loadtxt(paths[1], delimiter=',')
    X = np.vstack([X, X1])
    X2 = np.loadtxt(paths[2], delimiter=',')
    X = np.vstack([X, X2])
    return X


classifiers =[]
dir = glob.glob(pickles_dir + "*.pkl")
for path in dir:
    classifiers.append(joblib.load(path))
mv_clf = MajorityVoteClassifier(classifiers=classifiers, fitted=True)
X_test = collide_testset(testSet_paths)
proba = mv_clf.predict_proba(X_test)
with open("results_myller_v12.txt", 'w') as file:
    for prob in proba:
        file.write(str(prob[1]) + '\n')