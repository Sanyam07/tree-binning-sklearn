import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation, check_array
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

def check_is_column(X):
    """Checks that X is an n-dimentional array with 1 column. If X is a one
    dimentional array, it is reshaped into a the expected shape. If X contains
    more than 1 column, and error is thrown.

    Parameters
    ----------
    X : array-like
        Array to check, or reshape, into a single column.
    """
    if len(X.shape) == 1:
        # log a warning?
        return np.reshape(X, (-1, 1))
    elif X.shape[1] == 1:
        return X
    else:
        raise ValueError("X must be ndarray with 1 column.")


class BinnerMixin(object):
    """Mixin class for all binners."""


    def bin_value(self, val):
        """Return the bin a value should fall in based on thresholds_.

        Parameters
        ----------
        val : value to bin

        Returns
        -------
        binned_val : int
            Bin number
        """
        validation.check_is_fitted(self, 'thresholds_')
        result = None
        for i in range(len(self.thresholds_)):
            if i == 0:
                if val >=  val < self.thresholds_[i]:
                    result = i
            else:
                if val >= self.thresholds_[i-1] and val < self.thresholds_[i]:
                    result = i
        if result == None:
            return len(self.thresholds_)
        else:
            return result

    def transform(self, X, y=None):
        """Returns either a column of bin values resulting from binning
        X using thresholds_, or multiple indicator columns resulting from
        binning X using thresholds_ and then one-hot encoding.

        Parameters
        ----------
        X : array-like
            n-dimentional array with 1 column to transform into a bins.

        Returns
        -------
        binned : array-like
            Array of bin values. If object contains the one_hot attribure
            and it is equal to True, then returns multiple indicator columns.
        """
        validation.check_is_fitted(self, 'thresholds_')
        check_is_column(X)

        binned = np.array([self.bin_value(x) for x in X]).reshape(-1, 1)
        if getattr(self, 'one_hot', False):
            missing_levels = np.setdiff1d(
                np.arange(len(self.thresholds_+1)),
                np.unique(binned)
            )
            zero_col = np.repeat(0, X.shape[0])

            ohe = OneHotEncoder(sparse = False)
            binned = ohe.fit_transform(binned)
            for c in missing_levels:
                binned = np.insert(binned, c, zero_col.copy(), axis=1)
        return binned


class EqualFreqBinner(BaseEstimator, TransformerMixin, BinnerMixin):
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
    >>> X = np.arange(6)
    >>> X
    [0 1 2 3 4 5]
    >>> efb = EqualFreqBinner(3)
    >>> binned = efb.fit_transform(X)
    >>> binned
    [[ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 0.  0.  1.]]
    """

    def __init__(self, num_bins=10, one_hot=True):
        self.num_bins = num_bins
        self.one_hot = one_hot

        p_step = 100.0 / float(self.num_bins)
        self._percentiles = [100.0 - (p_step * i) for i in range(self.num_bins)]

    def fit(self, X, y=None):
        """Fits the binner by setting thresholds_ in a way that splits X into
        evenly populated bins.

        Parameters
        ----------
        X : array-like
            n-dimentional array with 1 column to fit the binner.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_is_column(X)
        self.thresholds_ = np.sort(np.percentile(X, self._percentiles))[:-1]
        return self

class EqualWidthBinner(BaseEstimator, TransformerMixin, BinnerMixin):
    """Bin a continuous variable into bins at uniformly-spaced intervals.

    Parameters
    ----------
    num_bins : how many bins are desired? Default = 10.


    Attributes
    ----------
    interval_ : spacing of cutpoints. Equal to range of X / num_bins
    thresholds_ : array of thresholds used to separate intervals


    Examples
    --------
    >>> X = np.arange(10)
    >>> X
    [0 1 2 3 4 5 6 7 8 9]
    >>> ewb = EqualWidthBinner(3)
    >>> binned = ewb.fit_transform(X)
    >>> binned
    [[ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  1.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 0.  0.  1.]]
    """

    def __init__(self, num_bins=10, one_hot=True):
        if num_bins <= 0:
            raise ValueError("num_bins = {0} cannot be less than or equal to zero.".format(num_bins))

        self.num_bins = num_bins
        self.one_hot = one_hot


    def fit(self, X, y=None):
        """Fits the binner by setting thresholds_ in a way that splits X into
        equally spaced intervals.

        Parameters
        ----------
        X : array-like
            n-dimentional array with 1 column to fit the binner.

        Returns
        -------
        self : object
            Returns self.
        """
        self.interval_ = float(np.max(X) - np.min(X)) / self.num_bins
        self.thresholds_ = [np.min(X) + (self.interval_ * bin) for bin in range(self.num_bins)][1:]

        return self

class TreeBinner(BaseEstimator, TransformerMixin, BinnerMixin):
    """Bins using a Decision Tree

    Parameters
    ----------
    max_depth : int (3 by default)
        Max depth of the tree used to bin the feature.

    one_hot : bool, optional (False by default)
        If True, returns the binned input feature a one hot encoded
        n-dimentional array.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([0, 0, 0, 1, 1, 1])

    >>> tr_bin = TreeBinner()
    >>> binned = tr_bin.fit_transform(X, y)
    >>> binned
    [[ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]]
    """

    def __init__(self, max_depth=3, one_hot=True):
        self.max_depth = max_depth
        self.one_hot = one_hot

    def fit(self, X, y):
        """Fits the binner by setting thresholds_ to the split points created
        by training an univariate decision tree.

        Parameters
        ----------
        X : array-like
            n-dimentional array with 1 column to fit the binner.

        y : array-like
            A one dimentional array of class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_is_column(X)
        self.decision_tree_ = DecisionTreeClassifier(max_depth=self.max_depth)
        self.decision_tree_.fit(X, y)
        self.thresholds_ = self.decision_tree_.tree_.threshold
        index = np.argwhere(self.thresholds_ == -2.0)
        self.thresholds_ = np.sort(np.delete(self.thresholds_, index))
        return self
