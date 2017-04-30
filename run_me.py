#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:17:40 2017

@author: mm94998
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
#import timeit as timeit
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#from plotting import *
from equal_width import *
from tree_binning import *
from equal_frequency import *

# ------------------------------------------------------------------------- #
#                   load datasets
# ------------------------------------------------------------------------- #
path = 'datasets/'

 # 3 class, 13 attributes, all continuous, no NAs
wine = np.loadtxt(path + 'wine.txt', delimiter = ',')

# 10 attributes (including ID), 214 instances
glass = np.loadtxt(path + 'glass.txt', delimiter = ',')
# 1st column in glass data set is an ID number; don't want
glass = glass[:, 1:]

iris = load_iris()
# ID features and label
iris_features = iris["data"]
iris_target = iris["target"]


heart = np.genfromtxt(path + 'heart.txt', delimiter = ',')
# heart has NaNs. Want to find the 6 cases w missing values and operate on
nas = np.argwhere(np.isnan(heart))
# not sure what imputation method makes sense; in the interest of progress...
# how about we do a complete case analysis and just remove these rows.
heart = np.delete(heart, nas[:,0], axis=0)
# check for NAs
print 'Heart data now contains', np.sum(np.isnan(heart)),'nans.'

# ------------------------------------------------------------------------- #
#               Scoring Function
# ------------------------------------------------------------------------- #

def getScore(binning_method, column_index, num_bins, X, y): 
    # do timing
    
    
    # perform binning on selected column and return transformed feature array
    if binning_method != None:
        
        # isolate column
        feat_to_bin = X[:, column_index]
        
        if binning_method == TreeBinner:
            
            binner = binning_method(one_hot = False)   
            binned_feat = binner.fit_transform(feat_to_bin.reshape(-1, 1), y.reshape(-1, 1))
            print 'binned_feat has shape', binned_feat.shape
        
        else:
            
            binner = binning_method(num_bins)   
            binned_feat = binner.fit_transform(feat_to_bin)            
            
        # delete original column, insert binned column into position
        X = np.delete(X, column_index, axis = 1)
        X = np.insert(X, column_index, binned_feat, axis=1)
        
    # want to take features and get all interaction terms
    x = PolynomialFeatures(interaction_only = True) # include_bias = should be F?
    x = x.fit_transform(X)
        
    # Split expanded data
    X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.01, random_state=234)

        
    # Initialize model
    model = LogisticRegression(penalty = 'l1', random_state = 123)

    # fit model object    
    model.fit(X_train, y_train) #Train X, Train Y
    
    # get mean cross validated score
    cv_scores = cross_val_score(model, x, y, cv = 5)
    
    # get test score
    test_score = model.score(X_test, y_test)
    
    
    print 'there are', x.shape[1],'features &', len(np.argwhere(model.coef_ != 0)), 'coefficients != 0.'
       
    return cv_scores, test_score #,runtime



# ------------------------------------------------------------------------- #
#           Let's see baseline accuracy for the data sets
# ------------------------------------------------------------------------- #

cv_scores, test_score = getScore(None, None, None, wine[:, 1:], wine[:, 0])
print "Wine CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Wine Test Score:", test_score, "\n"
# now with some binning...
cv_scores, test_score = getScore(EqualFreqBinner, 0, 25, wine[:, 1:], wine[:, 0])
print "Wine CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Wine Test Score:", test_score, "\n"

# ------------
cv_scores,test_score = getScore(None, None, None, glass[:, 0:-1], glass[:, -1])
print "Glass CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Glass Test Score:", test_score, "\n"

cv_scores, test_score = getScore(None, None, None, iris_features, iris_target)
print "Iris Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Iris Test Score:", test_score, "\n"

cv_scores, test_score = getScore(None, None, None, heart[:, 0:-1], heart[:, -1])
print "Heart Disease Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Heart Disease Test Score:", test_score, "\n"


# ------------------------------------------------------------------------- #
#                   Bin and Score Data 
# ------------------------------------------------------------------------- #
 # store name, with feature vector and target
datasets = {'Wine': (wine[:, 1:], wine[:, 0]), 
            'Glass': (glass[:, 0:-1], glass[:, -1]),
            'Iris': (iris_features, iris_target),
            'Heart': (heart[:, 0:-1], heart[:, -1])}

# initialize empty dicts to store scores
base = {}
equalwidth = {}
equalfreq = {}
tree = {}

for data in datasets:
    
    print 'Getting baseline score for', data
    
    # store mean CV score from baseline
    cv_scores, test_score = getScore(None, None, None, 
                                     datasets[data][0], datasets[data][1])
    base[data] = np.mean(cv_scores), test_score
        
    print 'Equal width binning on first feature column... ' # ------------ #
    
    column_index = 0
    num_bins = 4
    cv_scores, test_score = getScore(EqualWidthBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])
    
    equalwidth[data] = np.mean(cv_scores)
              
    print 'Equal frequency binning on first feature column... ' # -------- #
    
    cv_scores, test_score = getScore(EqualFreqBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])

    equalfreq[data] = np.mean(cv_scores)
             
    print 'TREE BINNING wooo on first feature column... ' # -------------- #
    
    cv_scores, test_score = getScore(TreeBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])

    tree[data] = np.mean(cv_scores)
    print "# --------------------- # \n"

print "baseline:", base.values(), "\n", "Equal Width:", equalwidth.values(), "\n",
"Equal Freq", equalfreq.values(),"\n", "and, Tree:",tree.values() 

# ------------------------------------------------------------------------- #
#                   Plot Scores    
# ------------------------------------------------------------------------- #   
baselines = [x[0] for x in base.values()]
equal_width = [x[0] for x in equalwidth.values()]
equal_freq = [x[0] for x in equalfreq.values()]
tree_bin = [x[0] for x in tree.values()]

index = np.arange(len(datasets))    
bar_width = .2

plt.bar(index - bar_width, baselines, bar_width, label = 'Baseline')
plt.bar(index, equal_width, bar_width, label = 'Equal Width')
plt.bar(index + bar_width, equal_freq, bar_width, label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree_bin, bar_width, label = 'Tree Binner')

plt.xticks(index + bar_width / 4, base.keys())
plt.legend()
plt.title('Scores by Dataset and Binning Method')
plt.xlabel('Dataset')
plt.ylabel('Mean Cross-Validated Accuracy')
plt.show()

"""
# add a table
table = plt.table(cellText=(bar1, bar2, bar3),
                  rowLabels=base.keys(),
                colLabels = ('Base', 'Equal Width', 'Equal Freq', 'Tree'),
                          loc='bottom')
table.set_fontsize(12)



# fake data
data = np.random.randn(100,3)
np.random.shuffle(data) 

num_bins = 5
col_ix = 0
tobin = data[:, col_ix]
equal_width = EqualWidthBinner(num_bins)   
equal_width.fit(tobin)
binned = equal_width.transform(tobin)
# delete old and insert new
data = np.delete(data, col_ix, axis = 1)
d = np.insert(data, col_ix, binned, axis=1)

plot_dist(data)

print 'Count of bins should equal', num_bins, 'and we have', len(np.unique(equal_width.transform(data)))
print 'Bins:', np.unique(equal_width.transform(data))
print 'Frequencies:',np.unique(equal_width.transform(data), return_counts=True)[1]
"""