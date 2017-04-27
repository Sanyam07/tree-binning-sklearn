import warnings
import numpy as np

from sklearn.utils.testing import assert_equal

from ..equal_frequency import EqualFreqBinner
from ..tree_binning import TreeBinner

from sklearn import datasets

rng = np.random.RandomState(0)

def test_equal_freq_binner():
    X_ = np.arange(25)
    np.random.shuffle(X_)

    efb = EqualFreqBinner(num_bins=5)
    X_bin = efb.fit_transform(X_)
    assert_equal(np.sum(X_bin == 0), 5)
    assert_equal(np.sum(X_bin == 1), 5)
    assert_equal(np.sum(X_bin == 2), 5)
    assert_equal(np.sum(X_bin == 3), 5)
    assert_equal(np.sum(X_bin == 4), 5)


def test_tree_binner():
    X_ = np.arange(25).reshape((25, 1))
    y_ = np.repeat(np.arange(5), 5, axis=0)

    efb = TreeBinner(max_depth=5)
    X_bin = efb.fit_transform(X_, y_)
    assert_equal(np.sum(X_bin == 0), 5)
    assert_equal(np.sum(X_bin == 1), 5)
    assert_equal(np.sum(X_bin == 2), 5)
    assert_equal(np.sum(X_bin == 3), 5)
    assert_equal(np.sum(X_bin == 4), 5)


test_equal_freq_binner()
test_tree_binner()
