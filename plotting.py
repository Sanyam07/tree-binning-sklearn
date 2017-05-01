#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:20:14 2017

@author: Martha Miller
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------------- Method by Column Plot : single dataset  --------------------------------------- #
def plotScoreByColumn(data, score_table, index_labels):
    
    baselines = score_table[1, :]
    equal_width = score_table[2, :]
    equal_freq = score_table[3, :]
    tree_bin = score_table[4, :]
    
    index = np.arange(len(index_labels))    
    
    plt.plot(index, baselines, 'sr--', lw = 4, label = 'Baseline')
    plt.plot(index, equal_width, 'sb--', lw = 2, label = 'Equal Width')
    plt.plot(index, equal_freq, 'sg--', lw = 2, label = 'Equal Frequency')
    plt.plot(index, tree_bin, 'sm--', lw = 2, label = 'Tree Binner')
    
    plt.xticks(index, index_labels)
    plt.legend(loc = "best")
    plt.title('%s: Scores by Column and Binning Method'%(data))
    plt.xlabel('Column Index')
    plt.ylabel('Test Accuracy')
    #plt.show()
    plt.savefig("Figures/%s_AllCols.pdf"%(data))

# ---------------- Runtime  ------------------------------------------ #
