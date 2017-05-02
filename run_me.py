#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:17:40 2017

@author: Martha Miller
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from scoring_funcs import *
#from plotting import *

# import scripts to do binning... not yet incorporated into sklearn
from binning import *
import time

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
print ('Heart data now contains', np.sum(np.isnan(heart)),'nans.')

cancer = np.genfromtxt(path + 'breast_cancer_wisc.txt', delimiter = ',')
# remove first column: ID number
cancer = cancer[:, 1:]
# Remove NaNs: Complete Cases
cancer = np.delete(cancer, np.argwhere(np.isnan(cancer)), axis=0)
print 'Breast Cancer data now contains', np.sum(np.isnan(cancer)),'nans.'

wine_cols = range(0, 13)
heart_cols =[0,3,4,7,9]
cancer_cols = np.arange(0,9)                                           
iris_cols =[0,1,2,3]
glass_cols = [0,1,2,3,4,5,6,7,8]
                                               

# ------------------------------------------------------------------------- #
#                   Bin and Score Data 
# ------------------------------------------------------------------------- #
 # store name, with feature vector and target
datasets = {'Wine': (wine[:, 1:], wine[:, 0], wine_cols), 
            'Glass': (glass[:, 0:-1], glass[:, -1], glass_cols),
            'Iris': (iris_features, iris_target, iris_cols),
            'Heart': (heart[:, 0:-1], heart[:, -1], heart_cols),
            'Cancer': (cancer[:, 0:-1], cancer[:, -1], cancer_cols)}

# initialize empty dicts to store scores
base = []
base_timings = []
cubic = []
cubic_timings = []
equalwidth = []
equalwidth_timings = []
equalfreq = []
equalfreq_timings = []
tree = []
tree_timings = []
datalist = []

for data in datasets:
    
    datalist.append(data)
    
    num_bins = 8
    print 'Getting baseline score for', data
    
    start = time.time()
    cv_scores = getScore(None, False, None, None, 
                                     datasets[data][0], datasets[data][1], None)
    base_timings.append(time.time() - start)
    mean = np.mean(cv_scores)
    base.append(mean)

    print 'Cubic expansion for', data
    start = time.time()
    cv_scores = getScore(None, False, None, None, datasets[data][0], datasets[data][1], 3)
    cubic_timings.append(time.time() - start)
    mean = np.mean(cv_scores)
    cubic.append(mean)

    print 'Equal width binning on feature column... ' # ------------ #
    print datasets[data][2]
    start = time.time()
    cv_scores = getScore(EqualWidthBinner, True, 
                                     datasets[data][2], num_bins, 
                                             datasets[data][0], datasets[data][1], None)
    equalwidth_timings.append(time.time() - start)
    
    mean = np.mean(cv_scores)
    equalwidth.append(mean) 

       
    print 'Equal frequency binning on feature column... ' # -------- #
    start = time.time()
    cv_scores = getScore(EqualFreqBinner, True, datasets[data][2], num_bins,
                                     datasets[data][0], datasets[data][1], None)
    
    equalfreq_timings.append(time.time() - start)
    
    mean = np.mean(cv_scores)
    equalfreq.append(mean)
    
    print 'TREE BINNING wooo on feature column... ' # -------------- #
    
    start = time.time()
    cv_scores = getScore(TreeBinner, True, datasets[data][2], num_bins,
                                     datasets[data][0], datasets[data][1], None)
    
    tree_timings.append(time.time() - start)
    
    mean = np.mean(cv_scores)
    tree.append(mean)

    print "# --------------------- # \n"

      
scores = np.concatenate((np.asarray(base).reshape(5,1), 
                         np.asarray(cubic).reshape(5,1),
                         np.asarray(equalwidth).reshape(5,1),
                         np.asarray(equalfreq).reshape(5,1),
                         np.asarray(tree).reshape(5,1)), axis = 1)


# ------------------------------------------------------------------------- #
#                   Plot Scores    
# ------------------------------------------------------------------------- #   
import matplotlib.pyplot as plt

index = np.arange(len(datasets))    
bar_width = .13

fig = plt.figure(figsize = (7, 4))
plt.bar(index - (bar_width *2), base, bar_width, color = 'crimson', label = 'Baseline')
plt.bar(index - bar_width, cubic, bar_width, color = 'k', label = 'Cubic')
plt.bar(index, equalwidth, bar_width,color = 'springgreen', label = 'Equal Width')
plt.bar(index + bar_width, equalfreq, bar_width, color = 'magenta', label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree, bar_width, color = 'mediumblue', label = 'Tree Binner')

plt.xticks(index + bar_width / 6, datalist)
#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.legend(loc="best")
plt.ylim(.5, 1)
plt.title('Scores: 8 Bins')
plt.xlabel('Dataset')
plt.ylabel('Mean 5-Fold Cross-Validated Accuracy')
#plt.tight_layout()
#plt.show()
plt.savefig("../Figures/AccuracyAll_8bins_has2nd.pdf")

base_timings = []
cubic = []
cubic_timings = []
equalwidth = []
equalwidth_timings = []
equalfreq = []
equalfreq_timings = []
tree = []
tree_timings = []
datalist = []

