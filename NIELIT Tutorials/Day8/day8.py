from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

data = load_boston()

X = data.data
y = data.target

ran = RandomForestRegressor()

ran.fit(X,y)

y_pred = ran.predict(X)

from sklearn.metrics import mean_squared_error

a1=mean_squared_error(y, y_pred)

print(a1)


import matplotlib.pyplot as plt

plt.scatter(y,y_pred)

plt.show()


#----------------SVR----------------------------

from sklearn.svm import SVR

svr = SVR()

svr.fit(X,y)

y_pred = svr.predict(X)

from sklearn.metrics import mean_squared_error

a2=mean_squared_error(y, y_pred)

print(a2)


import matplotlib.pyplot as plt

plt.scatter(y,y_pred)

plt.show()

#-----------------KNN-------------------------------


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()

knn.fit(X,y)

y_pred = knn.predict(X)

from sklearn.metrics import mean_squared_error

a3=mean_squared_error(y, y_pred)

print(a3)


import matplotlib.pyplot as plt

plt.scatter(y,y_pred)

plt.show()


#--------------Ridge---------------------------

from sklearn.linear_model import Ridge

rd = Ridge(0.5)

rd.fit(X,y)

y_pred = rd.predict(X)

from sklearn.metrics import mean_squared_error

a4=mean_squared_error(y, y_pred)

print(a4)


import matplotlib.pyplot as plt

plt.scatter(y,y_pred)

plt.show()


#----------------lasso--------------------------------


from sklearn.linear_model import Lasso

ls = Lasso()

ls.fit(X,y)

y_pred = ls.predict(X)

from sklearn.metrics import mean_squared_error

a5=mean_squared_error(y, y_pred)
print(a5)


import matplotlib.pyplot as plt

plt.scatter(y,y_pred)

plt.show()


#---------------------------------------------
'''
import numpy as np

x = np.array(a1,a2,a3,a4,a5)

plt.bar(x, height= [1,2,3,4,5])

plt.xticks(x+.5, ['a','b','c','d','e'])


from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import pandas as pd

models =[]

models.append(('RandomForest', RandomForestRegressor()))
models.append(('KNN', KNeighborsRegressor:()))
models.append(('Ridge', Ridge(0.5)))
models.append(('SVR', SVR()))
models.append(('Lasso', Lasso()))

#
scoring='accuracy'
results=[]
names =[]

for name,model in models:
    #kfold = KFold(n_splits=10, random_state = 7)
    v= mean_squared_error(y, y_pred)
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

'''   

