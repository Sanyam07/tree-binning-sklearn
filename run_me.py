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

from binning import *
import time
import pandas as pd 

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
base_test=[]
base_preproc_timings = []
base_score_timings = []

cubic = []
cubic_test = []
cubic_preproc_timings = []
cubic_score_timings = []

equalwidth = []
equalwidth_test = []
equalwidth_preproc_timings = []
equalwidth_score_timings = []

equalfreq = []
equalfreq_test = []
equalfreq_preproc_timings = []
equalfreq_score_timings = []

tree = []
tree_test = []
tree_preproc_timings = []
tree_score_timings = []

datalist = []


for data in datasets:
    
    datalist.append(data)
    
    num_bins = 4
    print 'Getting baseline score for', data
    
    cv_scores, test_score, datapreproc_time, score_time  = getScore(None, False, 
                                None, None, datasets[data][0], datasets[data][1], None)

    base_preproc_timings.append(datapreproc_time)
    base_score_timings.append(score_time)

    mean = np.mean(cv_scores)
    base.append(mean)
    base_test.append(test_score)



    print 'Cubic expansion for', data
    start = time.time()
    cv_scores, test_score, datapreproc_time, score_time = getScore(None, False, None, None, 
                                        datasets[data][0], datasets[data][1], 3)
    cubic_preproc_timings.append(datapreproc_time)
    cubic_score_timings.append(score_time)

    mean = np.mean(cv_scores)
    cubic.append(mean)
    cubic_test.append(mean)

    print 'Equal width binning on feature column... ' # ------------ #
    cv_scores, test_score, datapreproc_time, score_time = getScore(EqualWidthBinner, True, 
                                     datasets[data][2], num_bins, datasets[data][0], 
                                    datasets[data][1], None)
    equalwidth_preproc_timings.append(datapreproc_time)
    equalwidth_score_timings.append(score_time)

    mean = np.mean(cv_scores)
    equalwidth.append(mean)
    equalwidth_test.append(test_score)
       
    print 'Equal frequency binning on feature column... ' # -------- #
    start = time.time()
    cv_scores,test_score, datapreproc_time, score_time = getScore(EqualFreqBinner, 
                                    True, datasets[data][2], num_bins,
                                     datasets[data][0], datasets[data][1], None)

    equalfreq_preproc_timings.append(datapreproc_time)
    equalfreq_score_timings.append(score_time)
   
    mean = np.mean(cv_scores)
    equalfreq.append(mean)
    equalfreq_test.append(test_score)

    print 'TREE BINNING wooo on feature column... ' # -------------- #
    cv_scores, test_score, datapreproc_time, score_time = getScore(TreeBinner, 
                                    True, datasets[data][2], num_bins,
                                     datasets[data][0], datasets[data][1], None)
    tree_preproc_timings.append(datapreproc_time)
    tree_score_timings.append(score_time)
   
    mean = np.mean(cv_scores)
    tree.append(mean)
    tree_test.append(test_score)
    
    print "# --------------------- # \n"

      
scores = np.concatenate((np.asarray(datalist).reshape(5,1),
        np.asarray(base).reshape(5,1), 
                         np.asarray(cubic).reshape(5,1),
                         np.asarray(equalwidth).reshape(5,1),
                         np.asarray(equalfreq).reshape(5,1),
                         np.asarray(tree).reshape(5,1)), axis = 1)

test = np.concatenate((np.asarray(datalist).reshape(5,1),
        np.asarray(base_test).reshape(5,1), 
                         np.asarray(cubic_test).reshape(5,1),
                         np.asarray(equalwidth_test).reshape(5,1),
                         np.asarray(equalfreq_test).reshape(5,1),
                         np.asarray(tree_test).reshape(5,1)), axis = 1)

preproc_times = np.concatenate((np.asarray(datalist).reshape(5,1),
        np.asarray(base_preproc_timings).reshape(5,1), 
                         np.asarray(cubic_preproc_timings).reshape(5,1),
                         np.asarray(equalwidth_preproc_timings).reshape(5,1),
                         np.asarray(equalfreq_preproc_timings).reshape(5,1),
                         np.asarray(tree_preproc_timings).reshape(5,1)), axis = 1)

score_times = np.concatenate((np.asarray(datalist).reshape(5,1),
        np.asarray(base_score_timings).reshape(5,1), 
                         np.asarray(cubic_score_timings).reshape(5,1),
                         np.asarray(equalwidth_score_timings).reshape(5,1),
                         np.asarray(equalfreq_score_timings).reshape(5,1),
                         np.asarray(tree_score_timings).reshape(5,1)), axis = 1)

headers = ["Dataset", "Base", "Cubic", "EqualWidth", "EqualFreq","Tree"]

df = pd.DataFrame(scores)
df.columns = headers
df.to_csv("scores_%dbins.csv"%(num_bins), sep =',')

test = pd.DataFrame(test)
test.columns = headers
test.to_csv("Test_%dbins.csv"%(num_bins), sep =',')

timedf = pd.DataFrame(preproc_times)
timedf.columns = headers
timedf.to_csv("preproc_times_%dbins.csv"%(num_bins), sep=',')

timedf2 = pd.DataFrame(score_times)
timedf2.columns = headers
timedf2.to_csv("score_times_%dbins.csv"%(num_bins), sep=',')
# ------------------------------------------------------------------------- #
#                   Plot Scores    
# ------------------------------------------------------------------------- #   
index = np.arange(len(datasets))    
bar_width = .13

fig = plt.figure(figsize = (7, 4))
plt.bar(index - (bar_width *2), base, bar_width, color = 'crimson', label = 'Baseline')
plt.bar(index - bar_width, cubic, bar_width, color = 'k', label = 'Cubic')
plt.bar(index, equalwidth, bar_width,color = 'springgreen', label = 'Equal Width')
plt.bar(index + bar_width, equalfreq, bar_width, color = 'magenta', label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree, bar_width, color = 'mediumblue', label = 'Tree Binner')

plt.xticks(index + bar_width / 6, datalist)
plt.legend(loc="best")
plt.ylim(.5, 1)
plt.title('Validation Scores: %d Bins'%(num_bins))
plt.xlabel('Dataset')
plt.ylabel('Mean 5-Fold Cross-Validated Accuracy')
#plt.show()
plt.savefig("../Figures/ValAccuracy_%dbins.pdf"%(num_bins))

# Plot Test Score --------------------------------- #
fig = plt.figure(figsize = (7, 4))
plt.bar(index - (bar_width *2), base_test, bar_width, color = 'crimson', label = 'Baseline')
plt.bar(index - bar_width, cubic_test, bar_width, color = 'k', label = 'Cubic')
plt.bar(index, equalwidth_test, bar_width,color = 'springgreen', label = 'Equal Width')
plt.bar(index + bar_width, equalfreq_test, bar_width, color = 'magenta', label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree_test, bar_width, color = 'mediumblue', label = 'Tree Binner')

plt.xticks(index + bar_width / 6, datalist)
plt.legend(loc="best")
plt.ylim(.4, 1)
plt.title('Test Scores: %d Bins'%(num_bins))
plt.xlabel('Dataset')
plt.ylabel('Mean 5-Fold Cross-Validated Accuracy')
plt.savefig("../Figures/TestAccuracy_%dbins.pdf"%(num_bins))


# ---------------- Runtime  ------------------------------------------ #

#  Plot for TIME, get mean from 5-fold
base_timing = [x / 5 for x in base_preproc_timings]
cubic_timing= [x / 5 for x in cubic_preproc_timings]
equalwidth_timing = [x / 5 for x in equalwidth_preproc_timings]
equalfreq_timing = [x / 5 for x in equalfreq_preproc_timings]
tree_timing = [x / 5 for x in tree_preproc_timings]


plt.bar(index - (bar_width*2), base_timing, bar_width, color = 'crimson', label = 'Baseline')
plt.bar(index - bar_width, cubic_timing, bar_width, color = 'k', label = 'Cubic')
plt.bar(index, equalwidth_timing, bar_width, color = 'springgreen', label = 'Equal Width')
plt.bar(index + bar_width, equalfreq_timing, bar_width, color = 'magenta', label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree_timing, bar_width, color = 'mediumblue', label = 'Tree Binner')

plt.xticks(index + bar_width / 6, datalist)
plt.legend(loc="best")

plt.title('Average Preprocessing Time Required: %d Bins'%(num_bins))
plt.xlabel('Dataset')
plt.ylabel('Seconds')
plt.tight_layout()

#plt.show()
plt.savefig("../Figures/PreprocessingTime_%d.pdf"%(num_bins))


#  Plot for TIME, get mean from 5-fold
base_timings2 = [x / 5 for x in base_score_timings]
cubic_timings2= [x / 5 for x in cubic_score_timings]
equalwidth_timings2 = [x / 5 for x in equalwidth_score_timings]
equalfreq_timings2 = [x / 5 for x in equalfreq_score_timings]
tree_timings2 = [x / 5 for x in tree_score_timings]


plt.bar(index - (bar_width*2), base_timings2, bar_width, color = 'crimson', label = 'Baseline')
plt.bar(index - bar_width, cubic_timings2, bar_width, color = 'k', label = 'Cubic')
plt.bar(index, equalwidth_timings2, bar_width, color = 'springgreen', label = 'Equal Width')
plt.bar(index + bar_width, equalfreq_timings2, bar_width, color = 'magenta', label = 'Equal Frequency')
plt.bar(index + (bar_width*2), tree_timings2, bar_width, color = 'mediumblue', label = 'Tree Binner')

plt.xticks(index + bar_width / 6, datalist)
plt.legend(loc="best")
plt.title('Average Scoring Time Required: %d Bins'%(num_bins))
plt.xlabel('Dataset')
plt.ylabel('Seconds')
plt.tight_layout()

#plt.show()
plt.savefig("../Figures/ScoringTime_%d.pdf"%(num_bins))
