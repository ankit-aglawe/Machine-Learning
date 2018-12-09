
#data preprocessing 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('file_name.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# Column names List
Z = data.iloc[:,:-1].columns.tolist()  #for numpy array
Z = list(X.head(0))      #for pandas datafram


# Get Index 
k = data.iloc[:,:-1].index 


# Length of row and column
r = X.shape[0]
c = X.shape[1]


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()



# Encoding the data with loop
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le =LabelEncoder()
hotlist =[]
for i in range(X.shape[1]):
    if isinstance(X[1,i], str):
        X[:,i] = le.fit_transform(X[:,i])
        hotlist.append(i)

onehot = OneHotEncoder(categorical_features=hotlist)
X = onehot.fit_transform(X).toarray()




# Taking care of missing data with Imputer

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Taking care of missing data with fillna


# Comparing and plotting Classifier Accuracy
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import pandas as pd

kfold=KFold(10,random_state=7)
models=[]
models.append(("KNN",KNeighborsClassifier()))
models.append(("NB",GaussianNB()))
models.append(("LG",LogisticRegression()))
models.append(("Tree",DecisionTreeClassifier()))
models.append(("SVM",SVC()))

results=[]
names=[]
scoring='accuracy'
    
for name,model in models:
	kfold=KFold(n_splits=5,random_state=5) 
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