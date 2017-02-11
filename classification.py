 # importing the package
#SVC(kernel=”linear”) # building the classifier
#SVC(kernel=”rbf”) # building the classifier
#SVC(kernel=”poly”) # building the classifier
#classifier = KNeighborsClassifier(n_neighbors=3)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split



#X = iris.data[:, :2]
X_train, X_test, y_train, y_test = train_test_split(iris[['Sepal Length', 'Sepal Width']], iris['Petal Width'], test_size=0.3)
#X_train, X_test, y_train, y_test = train_test_split(df[['density', 'sulphates', 'residual_sugar']], df['high_quality'], test_size=0.3)

classifier1 = SVC(kernel= "linear")
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