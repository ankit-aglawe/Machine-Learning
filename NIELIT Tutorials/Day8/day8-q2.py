import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Death.txt', delimiter='\s+')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


from sklearn.ensemble import RandomForestRegressor


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

