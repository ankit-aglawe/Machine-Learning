import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bikerental.txt', delimiter='\s+')

X = data.iloc[:,[5,6]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


from sklearn.linear_model import LogisticRegression

logi = LogisticRegression()

logi.fit(X_train, y_train)

y_pred = logi.predict(X_test)

print(y_pred)
