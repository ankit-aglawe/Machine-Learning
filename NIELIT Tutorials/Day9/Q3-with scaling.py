import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Immunotherapy.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


#scaling

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()

X = sc.fit_transform(X)

#print(X)

#splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25 )


#knn
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test,y_pred))
print("accuracy score is ", accuracy_score(y_test,y_pred))




