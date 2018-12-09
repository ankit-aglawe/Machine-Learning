
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ex1.txt', delimiter=',')
#data = data.as_matrix()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(y_pred)

plt.scatter(y_test, y_pred)

plt.show()


#--------------------------------------------------------


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ex2.txt', delimiter=',')
#data = data.as_matrix()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(y_pred)

plt.scatter(y_test, y_pred)

plt.show()


#-----------------------------------------


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ex3.txt', delimiter=',')


X1 = data.iloc[:,[0]].values
X2 = data.iloc[:,[1]].values
ad = data.iloc[:,[2]].values

print(ad)
plt.scatter(X1, X2, c=ad)
plt.show()

print(type(ad))


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(y_pred)

plt.scatter(y_test, y_pred)

plt.show()



#-------------------------------------------------------------

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


data = pd.read_csv('titanic.csv', delimiter=',')
k = data.as_matrix()

data.fillna(data.mean(), inplace=True)

#print(data.head(20))

print(data.isnull().sum())

enc= preprocessing.OneHotEncoder()
le = preprocessing.LabelEncoder()
P = k[:,4]

le.fit(P)
Q = le.transform(P)
    
data.iloc[:,4] = Q

print(data.head(20))
      		
X = data.iloc[:,[2,4,5,6,7,9]].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


#lr = LinearRegression()

#lr.fit(X_train,y_train)

#y_pred = lr.predict(X_test)

print(y_pred)

plt.scatter(y_test, y_pred)

plt.show()



#-----------------------------------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

F = pd.read_csv("mining.csv", delimiter = ",")
F.fillna(F.mean(), inplace=True)

Q = F.as_matrix()

knn = KNeighborsClassifier(n_neighbors=2)
le = preprocessing.LabelEncoder()
le.fit(Q[:,1])
k = le.transform(Q[:,1])

Q[:,1] = k

X = Q[:,[1,3,4,5,6,7,8,9,10]]
y = Q[:, 0]

y=y.astype('int')

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X,y)

p = knn.predict([[12, 2807, 90.25, 0.346, 11.5, 20.23, 3.1, 1, 0.34]])

print(p)







