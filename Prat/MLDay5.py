
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def profit():
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()

    data = pd.read_csv('ex1.txt',delimiter=',')

    X = data.iloc[:,0:1].values
    y = data.iloc[:,-1].values

    print(X)
    print(y)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)
    print(y_pred)
    
    #Evaluation
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    print(mean_absolute_error(y_test,y_pred))
    print(mean_squared_error(y_test,y_pred))

    #plotting
    plt.scatter(X_test,y_pred)
    plt.show()


def price():
    data = pd.read_csv('ex2.txt',delimiter=',')

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    print(X.shape)
    print(y.shape)
    '''
    #scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(np.array([y]))
    '''
    print(X.shape)
    print(y.shape)

    #splitting the data

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    '''
    #scaling 
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train= sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test) 
    
    '''
    #regressor

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    regressor = Ridge(alpha=0.05,normalize=True)

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)
    print(y_pred)
    print(y_test)
    #evaluation

    from sklearn.metrics import mean_absolute_error,mean_squared_error

    print('Absolute error is {}'.format(mean_absolute_error(y_test,y_pred)))
    print('Squared Error is {}'.format(mean_squared_error(y_test,y_pred)))

    #plotting 

    plt.scatter(y_test,y_pred)
    plt.show()


def admission():
    data = pd.read_csv('ex3.txt',delimiter = ',')

    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    print(X.shape)
    print(y.shape)


    #splitting 
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    from sklearn.linear_model import LogisticRegression

    regressor = LogisticRegression()

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    print(y_pred)

    #evaluation

    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    #plotting

    plt.scatter(y_test,y_pred)
    plt.show()



def titanic():
    data = pd.read_csv('titanic.csv')
    print(data.shape)
    print(data.head)

    print(data.isnull().sum())

    print(data['Age'].head)

    data['Age'].fillna(round(data['Age'].mean()),inplace=True)
    print(data.isnull().sum()) 
    print(data['Age'].head)

    X = data.iloc[:,[0,2,4,5,6,7,9]].values
    y = data.loc[:,'Survived'].values

    print(X.shape)
    print(y.shape)
  
    #encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:,2]=le.fit_transform(X[:,2])
  

    from sklearn.preprocessing import OneHotEncoder
    oh = OneHotEncoder(categorical_features = [2])
    X = oh.fit_transform(X)

    #splitting
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression()

    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)

    #evaluation
    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    #plotting

    plt.scatter(y_test,y_pred)
    plt.show()

    
def minning():
    data = pd.read_csv('mining.csv')

    print(data.isnull().sum())

    X = data.iloc[:,3:11].values
    y = data.loc[:,'Facies'].values

    print(X.shape)
    print(y.shape)

    #splitting 

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
    print(confusion_matrix(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    #plotting

    plt.scatter(y_pred,y_test)
    plt.show()






def r51():
    from sklearn.linear_model import LinearRegression 
    lr = LinearRegression()

    data = pd.read_csv('ex1.txt',delimiter=',')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    #splitting 
    from sklearn.model_selection import train_test_split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_test)

    #plt.scatter(X_train,y_train)
    plt.scatter(y_pred,X_test)
    plt.show()

def r52():
    data = pd.read_csv('ex2.txt',delimiter=',')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    from sklearn.model_selection import train_test_split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor()

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    #evaluation

    from sklearn.metrics import mean_absolute_error,mean_squared_error
    print(mean_absolute_error(y_test,y_pred))
    print(mean_squared_error(y_test,y_pred))

    #plot

    plt.scatter(y_pred,y_test)
    plt.show()

def r53():
    import matplotlib.pyplot as plt 
    data = pd.read_csv('ex3.txt',delimiter=',')
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    from sklearn.model_selection import train_test_split 

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    from sklearn.tree import DecisionTreeClassifier
    regressor= DecisionTreeClassifier()

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
    
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

    plt.scatter(y_pred,y_test)
    plt.xlabel('abc')
    plt.ylabel('xyz')
    plt.show()

def r54():
    data = pd.read_csv('titanic.csv')
    print(data.shape)
    print(data.head)

    print(data.isnull().sum())

    data['Age'].fillna(round(data['Age'].mean()),inplace=True)

    print(data.isnull().sum())

    X = data.iloc[:,[0,2,4,5,6,7,9]].values
    y = data.loc[:,'Survived'].values

    print(X[5,:])

    from sklearn.preprocessing import LabelEncoder,OneHotEncoder

    le = LabelEncoder()
    X[:,2]=le.fit_transform(X[:,2])

    oh = OneHotEncoder(categorical_features=[2])
    X = oh.fit_transform(X).toarray()

    from sklearn.model_selection import train_test_split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression()

    regressor.fit(X_train ,y_train)

    y_pred = regressor.predict(X_test)

    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    plt.scatter(y_test,y_pred)
    plt.show()


