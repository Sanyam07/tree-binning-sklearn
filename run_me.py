#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:17:40 2017

@author: Martha Miller
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from scoring_funcs import *
from plotting import *

# import scripts to do binning... not yet incorporated into sklearn
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
# NaNs: how about we do a complete case analysis and just remove these rows.
heart = np.delete(heart, np.argwhere(np.isnan(heart)), axis=0)
print 'Heart data now contains', np.sum(np.isnan(heart)),'nans.'

cancer = np.genfromtxt(path + 'breast_cancer_wisc.txt', delimiter = ',')
# remove first column: ID number
cancer = cancer[:, 1:]
# Remove NaNs: Complete Cases
cancer = np.delete(cancer, np.argwhere(np.isnan(cancer)), axis=0)
print 'Breast Cancer data now contains', np.sum(np.isnan(cancer)),'nans.'

# ------------------------------------------------------------------------- #
#           Let's see baseline accuracy for the data sets
#               ... and also do some testing on TreeBinner
# ------------------------------------------------------------------------- #

cv_scores, test_score = getScore(None, None, None, wine[:, 1:], wine[:, 0])
print "Wine CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Wine Test Score:", test_score, "\n"

cv_scores, test_score = getScore(EqualWidthBinner, 0, 5, cancer[:, 0:-1], cancer[:, -1])
print "Cancer CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2)
print "Cancer Test Score:", test_score, "\n"

# now with some binning... Try each column in wine with ~ All Methods ~


wine_cols = range(0, 13)
wine_table = np.zeros((5, 13))
winetab = scoreColumnByMethod(wine[:, 1:], wine[:, 0], wine_cols, wine_table, 10)
plotScoreByColumn("Wine", winetab, winetab[0, :])

# try other data sets... all appropriate columns X each method
heart_cols =[0,3,4,7,9]
heart_table = np.zeros((5, 5))
hearttab = scoreColumnByMethod(heart[:, 0:-1], heart[:, -1], heart_cols, heart_table, 10)
plotScoreByColumn("Heart", hearttab, hearttab[0, :])

glass_cols = [0,1,2,3,4,5,6,7,8]
glass_table = np.zeros((5, 9))
glasstab = scoreColumnByMethod(glass[:, 0:-1], glass[:, -1], glass_cols, glass_table, 10)
plotScoreByColumn("Glass", glasstab, glasstab[0, :])

iris_cols =[0,1,2,3]
iris_table = np.zeros((5, 4))
iristab = scoreColumnByMethod( iris_features, iris_target, iris_cols, iris_table, 10)
plotScoreByColumn("Iris", iristab, iristab[0, :])

# ------------------------------------------------------------------------- #
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
            'Heart': (heart[:, 0:-1], heart[:, -1]),
            'Cancer': (cancer[:, 0:-1], cancer[:, -1])}

# initialize empty dicts to store scores
base = {}
equalwidth = {}
equalfreq = {}
tree = {}
cancer = {}

for data in datasets:
    
    print 'Getting baseline score for', data
    
    # store mean CV score from baseline
    cv_scores, test_score = getScore(None, None, None, 
                                     datasets[data][0], datasets[data][1])
    base[data] = np.mean(cv_scores), test_score
    table[row_idx, 0] = np.mean(cv_scores)
   
    print 'Equal width binning on feature column... ' # ------------ #
    
    column_index = 0
    num_bins = 10
    cv_scores, test_score = getScore(EqualWidthBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])
    
    equalwidth[data] = np.mean(cv_scores)
    table[row_idx, 1] = np.mean(cv_scores)
       
    print 'Equal frequency binning on feature column... ' # -------- #
    
    cv_scores, test_score = getScore(EqualFreqBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])

    equalfreq[data] = np.mean(cv_scores)
    table[row_idx, 2] = np.mean(cv_scores)
        
    print 'TREE BINNING wooo on feature column... ' # -------------- #
    
    cv_scores, test_score = getScore(TreeBinner, column_index, num_bins,
                                     datasets[data][0], datasets[data][1])

    tree[data] = np.mean(cv_scores)
    table[row_idx, 4] = np.mean(cv_scores)

    row_idx = row_idx + 1

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