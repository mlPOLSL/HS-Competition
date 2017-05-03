from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from EnsembleClassifier.MajorityVoteClassifier import MajorityVoteClassifier
import numpy as np


X = np.loadtxt("C:\\Users\\user\PycharmProjects\Hearthstone\\trainingSet_100k_v4.gz", delimiter=',')
X_train = X[:, 0:35]
y_train = X[:, 35]
clf1 = LogisticRegression(solver='liblinear', C=10, random_state=0)
clf3 = MLPClassifier(alpha=1, hidden_layer_sizes=(50, 30), learning_rate_init=0.0001)
clf2 = RandomForestClassifier()
pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
# clf_labels = ['Logistic Regression', 'MLP', 'RandomForestClassifier']
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3], weights=[0.4, 0.2, 0.4], vote='probability')
clf_labels = ['Majority Voting']
all_clf = [mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("Accuracy: %0.6f (+/- %0.6f) [%s]" % (scores.mean(), scores.std(), label))