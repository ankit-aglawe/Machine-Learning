
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ai1/My_Files/ML/city.csv")


data= data.as_matrix()


X = data[:,1:6]

y = data[:,-2]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)

l1 = LinearRegression()

l1.fit(X_train, y_train)

p = l1.predict(X_test)
print(p)

import matplotlib.pyplot as plt

plt.scatter(y_test,p)
#plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report 

knn= KNeighborsClassifier(n_neighbors=1)

#knn.fit(X_train, y_train)

#p= knn.predict(X_test)


'''
print(confusion_matrix(y_test,p))
print(accuracy_score(y_test,p))
print(classification_report(y_test,p))
'''

score = cross_val_score(knn,X,y,cv=5)
print(score)
