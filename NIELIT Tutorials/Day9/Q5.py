import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


dataset = datasets.load_iris()


alphas = np.array([1,0.1,0.01,0.001,0.0001,0])


knn = KNeighborsClassifier()
grid = GridSearchCV(estimator=knn, param_grid=dict(alpha=alphas))
grid.fit(dataset.data, dataset.target)
print(grid)


print(grid.best_score_)
print(grid.best_estimator_.alpha)
