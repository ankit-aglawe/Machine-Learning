import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bank.csv')

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

#splitting
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25)


#print(X)

#LogisticRegression

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)

print(y_pred_log)

plt.scatter(y_test, y_pred_log)

#plt.show()


#knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print(y_pred_knn)


#svm

from sklearn.svm import SVC

model = SVC(gamma=0.001)

model.fit(X_train, y_train)

y_pred_svm = model.predict(X_test)

print(y_pred_svm)

#nb

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print(y_pred_nb)

#comparison
from sklearn.model_selection import cross_val_score, KFold

kfold=KFold(10,random_state=7)
models=[]
models.append(("KNN",KNeighborsClassifier()))
models.append(("NB",GaussianNB()))
models.append(("LG",LogisticRegression()))

#models.append(("SVM",SVC()))
results=[]
names=[]
scoring='accuracy'
for name,model in models:
	kfold=KFold(n_splits=10,random_state=7) 
	v=cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
	results.append(v)
	names.append(name)
	print(name)
	print(v)
fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

