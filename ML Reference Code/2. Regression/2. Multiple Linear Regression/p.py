#data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features = [3])
X = onehot.fit_transform(X).toarray()

#splitting the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(y_test, y_pred)
plt.show()





