#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:25:21 2017

@author: Martha Miller
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from binning import *
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
import math

# ------------------------------------------------------------------------- #
#               Scoring Function
# ------------------------------------------------------------------------- #

def getScore(binning_method, one_hot, column_index, num_bins, X, y, degree): 
    
    print 'binning method is', binning_method, 'X is', X.shape

    start = time.time()

    ohe = OneHotEncoder(sparse = False)
    # want to OneHotEncode -BOTH- train and test, so operate on X
    all_cols = range(0, X.shape[1])
    
    if column_index != None:
        cols_to_OHE = [x for x in column_index if x not in all_cols]
    
        for col in cols_to_OHE:
            ohe.fit_transform(X[:, col].reshape(-1, 1))
        
        print 'Allcols:', all_cols, 'and cols_to_OHE:', cols_to_OHE, 'shape of X', X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3, 
                                                        random_state = 123)
    print 'Train shapes:', X_train.shape, y_train.shape
    og = X_train
    

    # proceed with binning
    if binning_method != None:
        
        # initialize new array to put binned features in
        newX = np.empty((X_train.shape[0],1))
        newXTest = np.empty((X_test.shape[0],1))
             
        # loop over each column in X
       
        for col in range(0, X_train.shape[1]):
            
            # if column in 'to bin' index... bin it
            if col in column_index:
                
                # isolate column
                feat_to_bin = X_train[:, col]
                feat_to_bin_test = X_test[:,col]
                print 'feat to bin', feat_to_bin.shape, 'for col', col
                
                # get binned_feat according to method
                if binning_method == TreeBinner:
                    depth = math.log(num_bins,2)
                    binner = binning_method(one_hot=one_hot, max_depth = depth)
                    binned_feat = binner.fit_transform(feat_to_bin.reshape(-1, 1), y_train.reshape(-1, 1))
                    binned_feat_test = binner.transform(feat_to_bin_test)
                else:
                    binner = binning_method(num_bins, one_hot=one_hot)   
                    binned_feat = binner.fit_transform(feat_to_bin)
                    binned_feat_test = binner.transform(feat_to_bin_test)

                
                print 'binned_feat has', binned_feat.shape
                    
                # move binned_feat to new array
                newX = np.concatenate((newX, binned_feat), axis = 1)
                newXTest = np.concatenate((newXTest, binned_feat_test), axis = 1)
                print 'newX has shape after addition of binned feat', newX.shape
               
            # if col not in 'to bin' index, move to new array
            else:
                print 'col not in "to bin"'

                newX = np.concatenate((newX, X_train[:, col].reshape(-1,1)), axis = 1)
                newXTest = np.concatenate((newXTest, X_test[:, col].reshape(-1,1)), axis = 1)
        
        # Outside of column loop, write over X and delete junk first col
        print('writing over newX', newX.shape, 'with X', X.shape)
        X_train = newX[:, 1:] 
        X_test = newXTest[:, 1:]
        
    else:
        print ('No Binning performed. X == original dims:', X_train.shape ==og.shape)

    # Add basis expansion if specified, else add interaction terms
    if degree != None:

        x = PolynomialFeatures(degree = degree) 
        X_train = x.fit_transform(X_train)
        X_test = x.fit_transform(X_test)

    else:     

        x = PolynomialFeatures(interaction_only = True) 
        X_train = x.fit_transform(X_train)
        X_test = x.fit_transform(X_test)
    
    print ('X after basis/interaction terms:', X_train.shape)    

    # get mean cross validated score
    print '# --------- Data processing done: getting cv scores --------- #'
    model = LogisticRegression(penalty='l1', random_state = 123)    
    cv_scores = cross_val_score(model, X_train, y_train, cv = 5, n_jobs = 5)
    
    print "# -------     Fitting model  ------------ #"
    
    model = LogisticRegression(penalty='l1', random_state = 123)
    params = {'C' : [0.1, 0.5, 1.0, 2.0, 10.0]}
    grid = GridSearchCV(model, params).fit(X_train, y_train)
    
    datapreproc_time = time.time() - start

    # Time to score the fitted model
    start = time.time()

    test_score = grid.score(X_test, y_test)
    
    score_time = time.time() - start
    
    print "test_score ...", test_score
           
    return cv_scores, test_score, datapreproc_time, score_time


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
