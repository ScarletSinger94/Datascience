#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:55:42 2017

@author: Singingking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

youtube_data = pd.read_csv('youtube_results1.csv',encoding='latin-1')
#youtube_data1 = pd.read_csv('youtube_results2.csv')
#plt.figure()
#hist1,edges1 = np.histogram(youtube_data.viewCount)
#plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])

print(youtube_data.corr())
plt.scatter(youtube_data.viewCount,youtube_data.likeCount)

y=youtube_data.likeCount
X=youtube_data.viewCount
X =sm.add_constant(X)
lr_model= sm.OLS(y,X).fit()
print(lr_model.summary())

#likecount= coeff* viewcount + const

#print(youtube_data1.corr())
#plt.scatter(youtube_data1.viewCount,youtube_data1.likeCount)
#
#y=youtube_data1.likeCount
#X=youtube_data1.viewCount
#X =sm.add_constant(X)
#lr_model= sm.OLS(y,X).fit()
#print(lr_model.summary())

#likecount= coeff* viewcount + const

X_prime = np.linspace(X.viewCount.min(),X.viewCount.max(),100)
X_prime = sm.add_constant(X_prime)
y_hat = lr_model.predict(X_prime)
plt.scatter(X.viewCount,y)
plt.xlabel("View Count")
plt.ylabel("Like Count")
plt.plot(X_prime[:,1],y_hat)