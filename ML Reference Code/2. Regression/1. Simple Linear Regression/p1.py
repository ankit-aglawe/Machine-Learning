# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:51:36 2018

@author: ankit
"""
#import the Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Importing the Dataset

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#splittting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#importing the LinearRegression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)


#predicting the dependent Value

y_pred = regressor.predict(X_test)

#plotting the training set

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experince vs salary- Training data')
plt.show()

#plotting the test Data

plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color='blue')
plt.title('experience vs salary- test data')
plt.xlabel('Experience')
plt.ylabel('salary')
plt.show()

