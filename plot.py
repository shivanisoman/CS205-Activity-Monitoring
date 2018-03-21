# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:37:40 2018

@author: shivu.soman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

arr = [[0.9377,0.9702,0.9944],
       [0.5623,0.6819,0.6716],
       [0.9203,0.9573,0.9900],
       [0.5572,0.5542,0.7597],
       [0.9445,0.9758,0.9944],
       [0.4966,0.6667,0.6078],
       [0.6345,0.7363,0.8760],
       [0.9224,0.9402,0.9794],
       [0.9142,0.8755,0.9974]]

labels = ['Random Forest','Naive Bayes','Decision Tree','Logistic Regression','KNN','NN','NN (scaled)','Gradient Boosting','LSTM']

df2 = pd.DataFrame(arr,columns = ['ACC','ACC + GYRO','ACC + ORI'])
ax = df2.plot(kind='bar')
ax.set_xlabel("Classifiers",fontsize=12)
ax.set_ylabel("Accuracy",fontsize=12)
ax.set_xticklabels(labels, fontsize = 7)
ax.legend(loc='lower right')
plt.savefig('foo.png',dpi = 720,bbox_inches = 'tight')

fig, axes = plt.subplots()
colors = ['red', 'tan', 'lime']
axes.hist(arr,n_bins = 9, normed=1, histtype='bar', color=colors, label=colors)
axes.legend(prop={'size': 10})
axes.set_title('bars with legend')

import numpy as np
a = [1,2,3]
np.median(a)

















