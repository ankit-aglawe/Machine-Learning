
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris=load_iris()

print(iris.data)
print(iris.target)

knn = KNeighborsClassifier(n_neighbors=1)
X=iris.data
y=iris.target

knn.fit(X,y)

p= knn.predict([[2,9,11,20]])

print(p)



print("--------------------------------------------------------------------------")
#-------------------------------------------------------------------------------------



X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.3)

knn.fit(X_train, y_train)

p1 = knn.predict(X_test)

print(confusion_matrix(y_test,p1))
print("accuracy score is ", accuracy_score(y_test,p1))


print("--------------------------------------------------------------------------")


#-------------------------------------------------------------------------------------



import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv("/home/ai1/My_Files/ML/Day2/Immunotherapy.csv", delimiter=",")

testData = pd.read_csv("/home/ai1/My_Files/ML/Day2/Immunotherapy.csv", delimiter=",")

train=trainData.as_matrix()
test= testData.as_matrix()

iknn = KNeighborsClassifier(n_neighbors=1)

print(train)
print(train.shape)
X=train[:,0:7]
y= train[:,7]

X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.3)


iknn.fit(X_train,y_train)

p2= iknn.predict(X_test)

print(p2)

print(confusion_matrix(y_test,p2))
print("accuracy score is ", accuracy_score(y_test,p2))

newlist =[]

for i in range(1,20):
  iknn = KNeighborsClassifier(n_neighbors=i)
  iknn.fit(X_train, y_train)
  p2= iknn.predict(X_test)
  a= accuracy_score(y_test,p2)
  newlist.append(a)

plt.plot(range(1,20),newlist)
plt.ylabel("accuracy score")



#----------------------------------------------------------------------------------------------
print("-------------------------------------------------------------------")

import pandas as pd


train1 = pd.read_csv("/home/ai1/My_Files/ML/ecoli.csv", delimiter=",")

test1 = pd.read_csv("/home/ai1/My_Files/ML/ecoli.csv", delimiter=",")

newtrain=train1.as_matrix()
newtest= test1.as_matrix()
jknn = KNeighborsClassifier(n_neighbors=1)

print(newtrain)
print(newtrain.shape)

X=newtrain[:,1:7]
y=newtrain[:,8]

X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.3)


jknn.fit(X_train,y_train)

p3= jknn.predict(X_test)

print(p3)

print(confusion_matrix(y_test,p3))
print("accuracy score is ", accuracy_score(y_test,p3))

newlist1 =[]

for i in range(1,50):
  jknn = KNeighborsClassifier(n_neighbors=i)
  jknn.fit(X_train,y_train)
  p3= jknn.predict(X_test)
  a= accuracy_score(y_test,p3)
  newlist.append(a)

plt.plot(range(1,50),newlist1)
plt.ylabel("accuracy score")
plt.show()


 
