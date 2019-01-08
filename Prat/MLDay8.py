import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def boston():
    from sklearn.datasets import load_boston
    boston = load_boston()

    X = boston.data
    y = boston.target

    models=[]

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge,Lasso 
    from sklearn.tree import DecisionTreeRegressor 
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn.svm import SVR 

    models.append(('LR',LinearRegression()))
    models.append(('Lasso',Lasso()))
    models.append(('Ridge',Ridge()))
    models.append(('Tree',DecisionTreeRegressor()))
    models.append(('Ensemble',RandomForestRegressor()))
    models.append(('SVR',SVR()))

    results=[]
    names=[]
    from sklearn.model_selection import KFold 
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    import math
    for name,model in models:
        kfold = KFold(n_splits=10,random_state=10)
        model.fit(X,y)
        y_pred=model.predict(X)
        a = math.sqrt(mean_squared_error(y,y_pred))
        results.append(a)
        names.append(name)
        print(name)
        print(a)
        print('========================')
    print(results)
  

    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
 
 
    y_pos = np.arange(len(names))
  
 
    plt.bar(y_pos, results, align='center', alpha=0.5)
    plt.xticks(y_pos, names)
    plt.ylabel('Usage')
    plt.title('Comaprision')
 
    plt.show()


def reg():
    data = pd.read_csv('Death.txt',delimiter='\s+')

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    from sklearn.ensemble import RandomForestRegressor 
    regressor = RandomForestRegressor()


    regressor.fit(X,y)

    y_pred = regressor.predict(X)

    from sklearn.metrics import mean_squared_error 
    print('RMSE value is {}'.format(np.sqrt(mean_squared_error(y_pred,y))))
    plt.scatter(y_pred,y)
    plt.show()

reg()