import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import style

style.use("ggplot")
iris = datasets.load_iris()
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["b.","r.","c.","y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]])

plt.scatter(centroids[:,0],centroids[:,1], marker="x", s=150, linewidths=3, zorder=10)
plt.show()