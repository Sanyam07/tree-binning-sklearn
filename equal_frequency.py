import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation
from sklearn.preprocessing import OneHotEncoder

def bin(val, thresholds):
    result = None
    for i in range(len(thresholds)):
        if i == 0:
            if val >=  val < thresholds[i]:
                result = i
        else:
            if val >= thresholds[i-1] and val < thresholds[i]:
                result = i
    if result == None:
        return len(thresholds)
    else:
        return result


class EqualFreqBinner(BaseEstimator, TransformerMixin):
    """Write documentation here.

    Parameters
    ----------
    num_bins : int (10 by default)
        Number of equally sized bins to split the feature into.

    one_hot : bool, optional (False by default)
        If True, returns the binned input feature a one hot encoded
        n-dimentional array.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(30)
    >>> X
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    >>> 

    """

    def __init__(self, num_bins=10, one_hot=True):
        self.num_bins = num_bins
        self.one_hot = one_hot

        p_step = 100.0 / float(self.num_bins)
        self._percentiles = [100.0 - (p_step * i) for i in range(self.num_bins)]

    def fit(self, X, y=None):
        self.thresholds_ = np.sort(np.percentile(X, self._percentiles))[:-1]
        return self

    def transform(self, X, y=None):
        validation.check_is_fitted(self, 'thresholds_')
        binned = np.array([bin(x, self.thresholds_) for x in X]).reshape(-1,1) #.reshape((len(X),1))
        
        if self.one_hot:
            ohe = OneHotEncoder(sparse = False)
            return ohe.fit_transform(binned.reshape(-1,1))

        return binned
        

# Test it out!

# a = np.arange(300) - 150
# np.random.shuffle(a)
# print a
#
# efb = EqualFreqBinner(num_bins=7, one_hot=True)
# b = efb.fit_transform(a)
# print b

# print
# print
#

# from sklearn.preprocessing import Binarizer
# r = np.random.rand(3, 3)
# print r
# bin = Binarizer(threshold=0.5)
# b = bin.fit_transform(r)
# print b
