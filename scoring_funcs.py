#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:25:21 2017

@author: Martha Miller
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from binning import *

# ------------------------------------------------------------------------- #
#               Scoring Function
# ------------------------------------------------------------------------- #

def getScore(binning_method, one_hot, column_index, num_bins, X, y, degree): 
    from sklearn.preprocessing import PolynomialFeatures
    
    print 'binning method is',binning_method
    og = X

    
    # proceed with binning
    if binning_method != None:
        # initialize new array to put binned features in
        newX = np.empty((X.shape[0],1))
        
        # loop over each column in X
        for col in range(0, X.shape[1]):
            
            # if column in 'to bin' index... bin it
            if col in column_index:
                
                # isolate column
                feat_to_bin = X[:, col]
                print 'feat to bin', feat_to_bin.shape, 'for col', col
                
                # get binned_feat according to method
                if binning_method == TreeBinner:
                    binner = binning_method(one_hot=one_hot)
                    binned_feat = binner.fit_transform(feat_to_bin.reshape(-1, 1), y.reshape(-1, 1))
                    
                else:
                    binner = binning_method(num_bins, one_hot=one_hot)   
                    binned_feat = binner.fit_transform(feat_to_bin)
                
                print 'binned_feat has', binned_feat.shape
                    
                # move binned_feat to new array
                newX = np.concatenate((newX, binned_feat), axis = 1)
                print 'newX has shape after addition of binned feat', newX.shape
               
            # if col not in 'to bin' index, move to new array
            else:
                print 'col not in "to bin"'
                newX = np.concatenate((newX, X[:, col].reshape(-1,1)), axis = 1)
            
        # Outside of column loop, newX is done so write over X
        print('writing over newX', newX.shape, 'with X', X.shape)
        X = newX[:, 1:] 
        
    else:
        print ('No Binning performed. X == original dims:', X.shape ==og.shape)
        
    # confirm what X is
    print('Out of binning loop, all feats binned. X has shape', X.shape) 
    
    if degree != None:
        # just get all interaction terms and expansion
        print('Basis expansion of degree', degree)
        x = PolynomialFeatures(degree = degree) 
        x = x.fit_transform(X)

    else:     
        print('just interaction terms')
        # just get all interaction terms
        x = PolynomialFeatures(interaction_only = True) 
        x = x.fit_transform(X)
    
    print ('X after basis/interaction terms:', x.shape)
      
    # Initialize model
    model = LogisticRegression(penalty = 'l1', random_state = 123)

    # get mean cross validated score
    print '# --------- Data processing done: getting cv scores --------- #'
    cv_scores = cross_val_score(model, x, y, cv = 5, n_jobs = 4)

           
    return cv_scores #test_score #,runtime

# ------------------------------------------------------------------------- #
#               Score Column by Method
# ------------------------------------------------------------------------- #

def scoreColumnByMethod(X, y, one_hot, cols, table, num_bins):

    for col in range(0, len(cols)):
        table[0, col] = int(cols[col])
        # Base
        cv_scores, test_score = getScore(None, False, None, None, X, y)
        table[1, col] = test_score
                  
        # Equal Width          
        cv_scores, test_score = getScore(EqualWidthBinner, one_hot, col, num_bins, X, y)
        print( "Equal Width  CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
        print("Equal Width  Test Score:", test_score, "\n")
        table[2, col] = test_score
        
        # Equal Freq
        cv_scores, test_score = getScore(EqualFreqBinner, one_hot, col, num_bins, X, y)
        print( "Equal Freq CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
        print ("Equal Freq Test Score:", test_score, "\n")
        table[3, col] = test_score
        
        # MDLP  
        cv_scores, test_score = getScore(MDLP, one_hot, col, num_bins, X, y)

        mdlp = MDLP()
        #X = mdlp.fit_transform(X, y)
        # Tree
        cv_scores, test_score = getScore(TreeBinner, one_hot, col, num_bins, X, y)
        print ("Tree CV Accuracy at: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
        print ("Tree Test Score:", test_score, "\n")
        table[4, col] = test_score
    
    
    return table
