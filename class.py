import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

from sklearn import datasets
from sklearn.svm import SVC # importing the package
iris = datasets.load_iris()

#df = pd.read_csv('wine.csv')
X_train, X_test, y_train, y_test = train_test_split(iris.data[:, :2],iris.target, test_size=0.3)
#X_train, X_test, y_train, y_test = train_test_split(df[['density', 'sulphates', 'residual_sugar']], df['high_quality'], test_size=0.3)
#classifier = KNeighborsClassifier(n_neighbors=3)
#classifier = SVC(kernel="linear")
classifier= SVC(kernel='rbf')
#classifier = SVC(kernel='poly')
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

correct = np.where(prediction==y_test, 1, 0).sum()
print(correct)
accuracy = correct/len(y_test)
print(accuracy)
results = []

for k in range(1, 51, 2):
    #classifier = KNeighborsClassifier(n_neighbors=k)
    #classifier = SVC(kernel="linear")
    classifier = SVC(kernel='rbf')
    #classifier = SVC(kernel='poly')
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    correct = np.where(prediction==y_test, 1, 0).sum()
    accuracy = correct/len(y_test)
    print(accuracy)
    results.append([k, accuracy])
    
results = pd.DataFrame(results, columns=["k", "accuracy"])

plt.plot(results.k, results.accuracy)
plt.show()