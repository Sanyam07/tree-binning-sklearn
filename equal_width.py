#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:48:46 2017

@author: Martha MIller

create class to implement equal-width binning
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
import matplotlib.pyplot as plt

# Here, bin determines how the returned variable is encoded.
# Do we want the variable levels to be [0,...num_bins] or [threshold min,..., max (by interval step)]?
def bin(data, val, thresholds):
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
    
    def __init__(self, num_bins=10, one_hot=False):
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
        #check_array(X, dtype=FLOAT, estimator=None) # need to check params meaning
        self.interval_ = (np.max(X) - np.min(X)) / (self.num_bins - 1)
        self.thresholds_ = [(np.min(X)+1) + (self.interval_ * bin) for bin in range(self.num_bins)]

        return self
    
    def transform(self, X, y=None):
        """Add doc
        """
        binned = np.array([bin(X, x, self.thresholds_) for x in X])
        if self.one_hot:
            ohe = OneHotEncoder()
            return ohe.fit_transform(binned).toarray()
        else:
            return binned
        return binned
   
# Hmm -- multiclass.unique_labels: useful here?
    #Helper function to extract an ordered array of unique labels 
    #from different formats of target.
'''
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
np.random.shuffle(data) 

num_bins = 50

equal_width = EqualWidthBinner(num_bins)   
equal_width.fit(data)


plot_dist(data)

print 'Count of bins should equal', num_bins, 'and we have', len(np.unique(equal_width.transform(data)))
print 'Bins:', np.unique(equal_width.transform(data))
print 'Frequencies:',np.unique(equal_width.transform(data), return_counts=True)[1]
'''