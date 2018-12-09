import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv("Social_Network_Ads.csv")

X = data.iloc[:,[1,2,3]].values
y = data.iloc[:,-1].values


#encoding 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

X[:,0] = le.fit_transform(X[:,0])

onehot = OneHotEncoder(categorical_features=[0])

X = onehot.fit_transform(X)

#print(X)


#splitting data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25)




from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred.shape)
print(X_test.shape)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



