import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


twitter_data = pd.read_csv('result_war.csv')
print (twitter_data.corr())
twitter_data_subjective = twitter_data[twitter_data['subjectivity']>0.5]
print(twitter_data_subjective.corr())
plt.scatter(twitter_data.polarity,twitter_data.subjectivity)


y=twitter_data.polarity
X=twitter_data.friends
X =sm.add_constant(X)
lr_model= sm.OLS(y,X).fit()
print(lr_model.summary())

#
#twitter_data1 = pd.read_csv('result_immigration.csv')
#print (twitter_data1.corr())
#twitter_data_subjective1 = twitter_data1[twitter_data1['subjectivity']>0.5]
#print(twitter_data_subjective1.corr())
#plt.scatter(twitter_data1.retwc,twitter_data1.polarity)
#
