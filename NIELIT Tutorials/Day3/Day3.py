

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ai1/My_Files/ML/Advertising.csv")

data= data.as_matrix()

X = data[:, [1,2,3]]

y = data[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)

l = LinearRegression()

l.fit(X_train, y_train)

p = l.predict(X_test)
print(p)

import matplotlib.pyplot as plt

plt.scatter(y_test,p)
#plt.show()




#-----------------------------------------------------------------------------------------------------



from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ai1/My_Files/ML/loan.csv")

data= data.as_matrix()

X = data[:, [1,2]]

y = data[:,-2]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.5)

l1 = LinearRegression()

l1.fit(X_train, y_train)

p = l1.predict(X_test)
print(p)

import matplotlib.pyplot as plt

plt.scatter(y_test,p)
#plt.show()


#----------------------------------------------------------------------------------------------------------


from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ai1/My_Files/ML/pimaindians.csv")

data= data.as_matrix()

X = data[:, 0:8]

y = data[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)

l1 = LogisticRegression()

l1.fit(X_train, y_train)

p = l1.predict(X_test)
print(p)

import matplotlib.pyplot as plt

plt.scatter(y_test,p)
#plt.show()


#------------------------------------------------------------------------------------------------------------



from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("/home/ai1/My_Files/ML/Blood.txt", delimiter="  ")

print(data)

data= data.as_matrix()

X = data[:,[1,2,3]]

y = data[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)

l1 = LinearRegression()

l1.fit(X_train, y_train)

p = l1.predict(X_test)
print(p)

import matplotlib.pyplot as plt

plt.scatter(y_test,p)
#plt.show()

from sklearn import metrics

print(metrics.mean_absolute_error(y_test,p))
print(metrics.mean_squared_error(y_test,p))
print(np.sqrt(mean_squared_error(y_error,p)))

#-------------------------------------------------------------------------------------------------------------




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

'''

#-----------------------------------------------------------------------------------------------------------


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris


iris = load_iris()

p= iris.data

q= iris.target

knn= KNeighborsClassifier(n_neighbors=1)



score = cross_val_score(knn,p, q, cv=5)
print(score)


knn.fit(p,q)
a= knn.predict([[3,5,4,2]])




print(a)

print(confusion_matrix(q,a))
print(accuracy_score(q,a))
print(classification_report(q,a))

