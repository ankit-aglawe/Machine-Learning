from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
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

'''
#feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y).reshape(-1,1)

'''
#comparison
from sklearn.model_selection import cross_val_score, KFold

kfold=KFold(n_splits=10,random_state=7)
models=[]
models.append(("KNN",KNeighborsClassifier()))
models.append(("NB",GaussianNB()))
models.append(("LG",LogisticRegression()))
models.append(('tree', DecisionTreeClassifier()))
#models.append(("SVM",SVC()))
results=[]
names=[]

scoring='accuracy'


for name,model in models:
	kfold=KFold(n_splits=10,random_state=7) 
	v=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
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

