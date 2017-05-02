#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:48:46 2017

@author: Martha MIller

create class to implement equal-width binning
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
import matplotlib.pyplot as plt

def bin(val, thresholds):
    result = None
    for i in range(len(thresholds)):

        if i == 0:
            if val >= val < thresholds[i]:
                result = i
        else:
            if val >= thresholds[i-1] and val < thresholds[i]:
                result = i

    return result


class EqualWidthBinner(BaseEstimator, TransformerMixin):
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
    >>> from equal-width import *
    >>> 
    >>> binned = EqualWidthBinner()
    >>> binned.thresholds_
    array([]) ....... work on this later
    
    """
    
    def __init__(self, num_bins=10, one_hot=True):
        if num_bins <= 0:
            raise ValueError("num_bins = {0} cannot be less than or equal to zero.".format(num_bins))        
        
        self.num_bins = num_bins
        self.one_hot = one_hot

           
    def fit(self, X, y=None):
        """Fit equal-width binner
        
        Parameters
        ----------
        x : array-like of shape (n_samples,)
            feature vector.
            
        Returns
        -------
        self : returns an instance of self.
        """
        self.interval_ = (np.max(X) - np.min(X)) / (self.num_bins - 1)
        self.thresholds_ = [(np.min(X)+1) + (self.interval_ * bin) for bin in range(self.num_bins)]

        return self
    
    def transform(self, X, y=None):
        """Add doc
        """
        binned = np.array([bin(x, self.thresholds_) for x in X]).reshape(-1,1) #.reshape((len(X),1))
        if self.one_hot:
            ohe = OneHotEncoder(sparse = False)
            return np.array(ohe.fit_transform(binned.reshape(-1,1)))
        else:
            return binned
        return binned
   
"""
# ------- Testing ----------------------- # 
def plot_dist(data):
    # look at plot of transformed values
    plt.hist(equal_width.transform(data))
    plt.title("Histogram of Binned Data")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()

# fake data
data = np.random.randn(1000,1)

# Look at behavior of function when OHE is turned on/off
# want to see same encodings in feat x, 
# OHE == off we get one col with bins, OHE==on we get num_bins columns with 0,1s
num_bins = 6
OHE = EqualWidthBinner(num_bins, one_hot = True) 
notOHE = EqualWidthBinner(num_bins, one_hot = False) 
dat = OHE.fit_transform(data)
da = notOHE.fit_transform(data)

print 'dimensions of OHE should be', num_bins, num_bins == dat.shape[1]
print 'Not OHE, bins should equal', num_bins, num_bins == len(np.unique(da))
print 'Not OHE Bin Encodings:', np.unique(da)
freqs = np.unique(da, return_counts=True)[1]
print 'Frequencies in bins:', freqs
print 'First col in OHE should have sum of frequencies from not OHE:', np.sum(dat[:,0]) == freqs[0]
print '3rd col in OHE should have sum of frequencies from notOHE:', np.sum(dat[:,3]) == freqs[3]
"""