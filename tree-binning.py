import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets

def bin(val, thresholds):
    result = -1
    for i in range(len(thresholds)):
        if val >= thresholds[i]:
            result = i
    if result == -1:
        return len(thresholds)
    else:
        return result

class TreeBinner(BaseEstimator, TransformerMixin):
    """Bins using a Decision Tree
    """

    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        # TODO should we check that X is 1 column?
        self.decision_tree_ = DecisionTreeClassifier(max_depth=self.max_depth)
        self.decision_tree_.fit(X, y)
        self.thresholds_ = self.decision_tree_.tree_.threshold
        index = np.argwhere(self.thresholds_==-2.0)
        self.thresholds_ = np.sort(np.delete(self.thresholds_, index))
        return self

    def transform(self, X, y=None):
        validation.check_is_fitted(self, 'thresholds_')
        return np.array([bin(x, self.thresholds_) for x in X])


iris = datasets.load_iris()

X = iris.data[:,0]
print np.min(X), np.max(X)
X.shape = (150, 1)

tb = TreeBinner()
f1 = tb.fit_transform(X, iris.target)
print 'Thresholds', tb.thresholds_
print f1
