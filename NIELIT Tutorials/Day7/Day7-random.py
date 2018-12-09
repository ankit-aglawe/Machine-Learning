from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd


#url="https://github.com/mhyhre/mlcourse_open/blob/master/data/bikes_rent.csv"

data=pd.read_csv('bikerental.txt', delimiter='\s+')

X=data.iloc[:,[5,6]].values
y=data.iloc[:,-1].values

print(X)
print(y)

'''
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
'''
models =[]

models.append(('tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression()))

scoring='accuracy'
results=[]
names =[]

for name,model in models:
    kfold = KFold(n_splits=10, random_state = 7)
    v= cross_val_score(model, X, y, cv=kfold, scoring = scoring)
    results.append(v)
    names.append(name)
    print(name)
    print(v)

import matplotlib.pyplot as plt

fig=plt.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

    
    
