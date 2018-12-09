#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:48:09 2018

@author: ankit
"""
#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

'''
#encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

onehot = OneHotEncoder(categorical_features=[1])
X = onehot.fit_transform(X).toarray()


#imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
X[:,1] = imputer.fit_transform(X[:,1]) 

'''


#linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
poly_x = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_x, y)



#linear plotting 

plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X))
plt.title('Linear Regression')
plt.show()

#polynomial plottting

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_x))
plt.title('Polynomial Regression')
plt.show()


#linear predict
lin_reg.predict(6.5)


#polynomial predict
lin_reg_2.predict(poly_reg.fit_transform(6.5))





