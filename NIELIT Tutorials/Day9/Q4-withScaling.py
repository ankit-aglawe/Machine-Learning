import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('banking.csv')


X = data.iloc[:,[0,1,3,5,6,10,11,12,13,15,16,17,18,19]].values
y = data.iloc[:,-1].values

#print(X)


#encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])
X[:,3] = le.fit_transform(X[:,3])
X[:,4] = le.fit_transform(X[:,4])

onehot = OneHotEncoder(categorical_features=[1,2,3,4])
X = onehot.fit_transform(X).toarray()


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X, y)

y_pred = classifier.predict(X)


from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y,y_pred))
print("accuracy score is ", accuracy_score(y,y_pred))



