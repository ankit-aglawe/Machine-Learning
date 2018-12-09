import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf.fit(X, y)

y_pred = rf.predict(X)

print(y_pred)

plt.scatter(X, y, color='red')
plt.plot(X, y_pred,  color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Posion Level')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rf.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()