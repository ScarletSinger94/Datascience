import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from matplotlib import style

news = pd.read_csv('OnlineNewsPopularity.csv', index_col=0)
##news.columns
##news.columns= [x.strip() for x in news.columns]
#print("abs_title_sentiment_polarity",np.corrcoef(news.shares,news.abs_title_sentiment_polarity)[0,1])
#print("unique",np.corrcoef(news.shares,news.n_unique_tokens)[0,1])
#print("subjectivity",np.corrcoef(news.shares,news.title_subjectivity)[0,1])
#print("nonstop unique",np.corrcoef(news.shares,news.n_non_stop_unique_tokens)[0,1])
#fig1=plt.scatter(news.shares,news.abs_title_sentiment_polarity)
#plt.xlabel('shares', fontsize=18)
#plt.ylabel('abs_title_sentiment', fontsize=16)
#
#X=news.shares
#y=news.abs_title_sentiment_polarity
#
#X =sm.add_constant(X) 
#
#lr_model= sm.OLS(y,X).fit()
#print(lr_model.summary())
##report the above model part 1
#
#weekend=news[news.is_weekend == 1]
#print("abs_title_sentiment_polarity",np.corrcoef(weekend.shares,weekend.abs_title_sentiment_polarity)[0,1])
#print("unique",np.corrcoef(weekend.shares,weekend.n_unique_tokens)[0,1])
#print("subjectivity",np.corrcoef(weekend.shares,weekend.title_subjectivity)[0,1])
#print("nonstop unique",np.corrcoef(weekend.shares,weekend.n_non_stop_unique_tokens)[0,1])
#fig2=plt.scatter(weekend.shares,weekend.abs_title_sentiment_polarity)
#plt.xlabel('shares', fontsize=18)
#plt.ylabel('abs_title_sentiment', fontsize=16)
#
#X=weekend.shares
#y=weekend.abs_title_sentiment_polarity
#
#X =sm.add_constant(X) 
#
#lr_model= sm.OLS(y,X).fit()
#print(lr_model.summary())
##end part2
#weekday=news[news.is_weekend == 0]
#print("abs_title_sentiment_polarity",np.corrcoef(weekday.shares,weekday.abs_title_sentiment_polarity)[0,1])
#print("unique",np.corrcoef(weekday.shares,weekday.n_unique_tokens)[0,1])
#print("subjectivity",np.corrcoef(weekday.shares,weekday.title_subjectivity)[0,1])
#print("nonstop unique",np.corrcoef(weekday.shares,weekday.n_non_stop_unique_tokens)[0,1])
#fig3=plt.scatter(weekday.shares,weekday.abs_title_sentiment_polarity)
#plt.xlabel('shares', fontsize=18)
#plt.ylabel('abs_title_sentiment', fontsize=16)
#
#X=weekday.shares
#y=weekday.abs_title_sentiment_polarity
#
#X =sm.add_constant(X) 
#
#lr_model= sm.OLS(y,X).fit()
#print(lr_model.summary())
##end part 3
#
##p.4
#
##part5
#
#news_low = news[news.shares<1400]
#news_high = news[news.shares>=1400]
#
#
#fig4=plt.scatter(news_low.shares, news_low.abs_title_sentiment_polarity)
#plt.show(fig4)
#plt.xlabel('shares low', fontsize=18)
#plt.ylabel('abs_title_sentiment low', fontsize=16)
#fig5=plt.scatter(news_high.shares, news_high.abs_title_sentiment_polarity)
#plt.show(fig5)
#plt.xlabel('shares high', fontsize=18)
#plt.ylabel('abs_title_sentiment high', fontsize=16)
##style.use("ggplot")
#Z = news.data[:, news_clustering.iloc[i,0], news_clustering.iloc[i,1] ]
#
#kmeans = KMeans(n_clusters=1400)
#kmeans.fit(Z)
#
#centroids = kmeans.cluster_centers_
#labels = kmeans.labels_
#
#print(centroids)
#print(labels)
#
#colors = ["b.","r.","c.","y."]
#
#for i in range(len(X)):
#    plt.plot(X[i][0], X[i][1], colors[labels[i]])
#
#plt.scatter(centroids[:,0],centroids[:,1], marker="x", s=150, linewidths=3, zorder=10)
#plt.show()
#
##p6
##p7
X_train, X_test, y_train, y_test = train_test_split(news[['title_subjectivity', 'abs_title_sentiment_polarity']], news['shares'], test_size=0.3)
classifier1 = KNeighborsClassifier(n_neighbors=100)
#classifier1 = SVC(kernel= "linear")
classifier1.fit(X_train, y_train)

prediction = classifier1.predict(X_test)

correct = np.where(prediction==y_test, 1, 0).sum()
print(correct)

results = []

for k in range(1, 51, 2):
    classifier1 = SVC(kernel="linear")
    classifier1.fit(X_train, y_train)
    prediction = classifier1.predict(X_test)
    correct = np.where(prediction==y_test, 1, 0).sum()
    accuracy = correct/len(y_test)
    results.append([k, accuracy])
    
results = pd.DataFrame(results, columns=["k", "accuracy"])

plt.plot(results.k, results.accuracy)
plt.show()

news_low_extracted['pop'] = 0
news_high_extracted['pop'] = 1
frames = [news_low_extracted, news_high_extracted]
newsdata_classification = pd.concat(frames)
X = news_classification.iloc[:,0:2] 
y = news_classification['pop']