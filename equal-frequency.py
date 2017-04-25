import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation
from sklearn.preprocessing import Binarizer

def bin(val, thresholds):
    result = -1
    for i in range(len(thresholds)):
        if val >= thresholds[i]:
            result = i
    if result == -1:
        return len(thresholds)
    else:
        return result

class EqualFreqBinner(BaseEstimator, TransformerMixin):
    """Write documentation here.

    Parameters
    ----------
    num_bins : int (10 by default)
        Number of equally sized bins to split the feature into.
    """

    def __init__(self, num_bins=10):
        self.num_bins = num_bins

        p_step = 100.0 / float(self.num_bins)
        self._percentiles = [100.0 - (p_step * i) for i in range(self.num_bins)]

    def fit(self, X, y=None):
        self.thresholds_ = np.sort(np.percentile(X, self._percentiles))
        return self

    def transform(self, X, y=None):
        validation.check_is_fitted(self, 'thresholds_')
        return np.array([bin(x, self.thresholds_) for x in X])


# Test it out!

a = np.arange(300) - 150
np.random.shuffle(a)
print a

efb = EqualFreqBinner(num_bins=7)
b = efb.fit_transform(a)
print b

# print
# print
#
# r = np.random.rand(3, 3)
# print r
# bin = Binarizer(threshold=0.5)
# b = bin.fit_transform(r)
# print b
