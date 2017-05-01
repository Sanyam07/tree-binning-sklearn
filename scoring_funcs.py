#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:25:21 2017

@author: Martha Miller
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
#import timeit as timeit
#from sklearn.preprocessing import PolynomialFeatures

from equal_width import *
from tree_binning import *
from equal_frequency import *
# ------------------------------------------------------------------------- #
#               Scoring Function
# ------------------------------------------------------------------------- #

def getScore(binning_method, one_hot, column_index, num_bins, X, y): 
    from sklearn.preprocessing import PolynomialFeatures

    # do timing
    
    print 'Operating on X with shape', X.shape, 'with y of dim', y.shape
    # perform binning on selected column and return transformed feature array
    if binning_method != None:
        
        # isolate column
        feat_to_bin = X[:, column_index]
        
        if binning_method == TreeBinner:
            
            binner = binning_method(one_hot=one_hot)
            binned_feat = binner.fit_transform(feat_to_bin.reshape(-1, 1), y.reshape(-1, 1))
            print 'Binned_feat in Treebinner', binned_feat.shape
        else:
            
            binner = binning_method(num_bins, one_hot=one_hot)   
            binned_feat = binner.fit_transform(feat_to_bin)            
            print 'Binnedfeat in binner is', binned_feat.shape
        
        print 'binnedfeat shape:',binned_feat.shape


        # if OneHotEncoding, need to insert several columns, so make index range
        
        if one_hot:

            X = np.concatenate((X[:,0:-column_index], binned_feat,X[:,column_index:]), axis=1)
        else:
            # delete original column
            X = np.delete(X, column_index, axis = 1)
            # insert binned column(s) into position    
            X = np.insert(X, column_index, binned_feat, axis=1)

    # want to take features and get all interaction terms
    x = PolynomialFeatures(interaction_only = True) # include_bias = should be F?
    x = x.fit_transform(X)
    print 'X, with binned feat, has shape', X.shape, 'then transformed to include interaction terms, has shape', x.shape
        
    # Split expanded data
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.40, random_state=234)
        
    # Initialize model
    model = LogisticRegression(penalty = 'l1', random_state = 123)

    # fit model object    
    model.fit(X_train, y_train) #Train X, Train Y
    
    # get mean cross validated score
    cv_scores = cross_val_score(model, x, y, cv = 5, n_jobs = 4)
    
    # get test score
    test_score = model.score(X_test, y_test)
    
    print 'there are', x.shape[1],'features &', len(np.argwhere(model.coef_ != 0)), 'coefficients != 0.'
       
    return cv_scores, test_score #,runtime



# ------------------------------------------------------------------------- #
#               Score Column by Method
# ------------------------------------------------------------------------- #

def scoreColumnByMethod(X, y, cols, table, num_bins):

    for col in range(0, len(cols)):
        print col
        table[0, col] = int(cols[col])
        # Base
        cv_scores, test_score = getScore(None, None, None, X, y)
        table[1, col] = test_score
                  
        # Equal Width          
        cv_scores, test_score = getScore(EqualWidthBinner, col, num_bins, X, y)
        print "Equal Width  CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
        print "Equal Width  Test Score:", test_score, "\n"
        table[2, col] = test_score
        
        # Equal Freq
        cv_scores, test_score = getScore(EqualFreqBinner, col, num_bins, X, y)
        print "Equal Freq CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
        print "Equal Freq Test Score:", test_score, "\n"
        table[3, col] = test_score
        
        # Tree
        cv_scores, test_score = getScore(TreeBinner, col, num_bins, X, y)
        print "Tree CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
        print "Tree Test Score:", test_score, "\n"
        table[4, col] = test_score
    
    
    return table
