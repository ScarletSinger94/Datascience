#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:07:22 2017

@author: Singingking
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from pandas.stats.api import ols
from matplotlib import style
twitter_data = pd.read_csv('results_comey.csv')

print(twitter_data[twitter_data.columns[1:]].corr()['polarity'])
#greatest value is that which correlates with weight
y=twitter_data.subjectivity
X=twitter_data.polarity
X =sm.add_constant(X)
lr_model= sm.OLS(y,X).fit()
print(lr_model.summary())
plt.scatter(twitter_data.polarity,twitter_data.subjectivity)

#
#z=twitter_data.subjectivity
#Y=twitter_data[['polarity','followers']]
#Y =sm.add_constant(Y)
#lr_model= sm.OLS(z,Y).fit()
#print(lr_model.summary())
