import warnings
import numpy as np

from sklearn.utils.testing import assert_equal

from ..binning import EqualFreqBinner, EqualWidthBinner, TreeBinner

from sklearn import datasets

rng = np.random.RandomState(0)

def test_equal_freq_binner():
    X_ = np.arange(25).reshape(-1, 1)
    np.random.shuffle(X_)

    efb = EqualFreqBinner(num_bins=5, one_hot=False)
    X_bin = efb.fit_transform(X_)
    assert_equal(np.sum(X_bin == 0), 5)
    assert_equal(np.sum(X_bin == 1), 5)
    assert_equal(np.sum(X_bin == 2), 5)
    assert_equal(np.sum(X_bin == 3), 5)
    assert_equal(np.sum(X_bin == 4), 5)

    efb = EqualFreqBinner(num_bins=5)
    X_bin = efb.fit_transform(X_)
    assert_equal(X_bin.shape, (25,5))
    assert_equal(np.sum(X_bin), 25)

def test_equal_width_binner():
    X_ = np.arange(25).reshape(-1, 1)
    np.random.shuffle(X_)

    tb = EqualWidthBinner(num_bins=5, one_hot=False)
    X_bin = tb.fit_transform(X_)
    assert_equal(np.sum(X_bin == 0), 5)
    assert_equal(np.sum(X_bin == 1), 5)
    assert_equal(np.sum(X_bin == 2), 5)
    assert_equal(np.sum(X_bin == 3), 5)
    assert_equal(np.sum(X_bin == 4), 5)

    tb = EqualWidthBinner(num_bins=5)
    X_bin = tb.fit_transform(X_)
    assert_equal(X_bin.shape, (25,5))
    assert_equal(np.sum(X_bin), 25)

def test_tree_binner():
    X_ = np.arange(25).reshape((25, 1))
    y_ = np.repeat(np.arange(5), 5, axis=0)

    tb = TreeBinner(max_depth=5, one_hot=False)
    X_bin = tb.fit_transform(X_, y_)
    assert_equal(np.sum(X_bin == 0), 5)
    assert_equal(np.sum(X_bin == 1), 5)
    assert_equal(np.sum(X_bin == 2), 5)
    assert_equal(np.sum(X_bin == 3), 5)
    assert_equal(np.sum(X_bin == 4), 5)

    tb = TreeBinner(max_depth=5)
    X_bin = tb.fit_transform(X_, y_)
    assert_equal(X_bin.shape, (25,5))
    assert_equal(np.sum(X_bin), 25)

test_equal_freq_binner()
test_equal_width_binner()
test_tree_binner()
