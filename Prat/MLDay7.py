import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def iris():
    from sklearn.datasets import load_iris
    iris = load_iris()

    X = iris.data 
    y = iris.target 

    #data import
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score, KFold
    
    models=[]

    models.append(('Tree',DecisionTreeClassifier()))
    models.append(('LR',LogisticRegression()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('SVC',SVC()))
    models.append(('NB',GaussianNB()))

    scoring = 'accuracy'
    print(models)

    results=[]
    names=[]

    for name,model in models:
        kfold = KFold(n_splits=10,random_state=10)
        a = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(a)
        names.append(name)
        print(name)
        print(a)
        print('==============================')

    fig=plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax=fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def pima():

    url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    data = pd.read_csv(url)
    
    X = data.iloc[:,0:8].values
    y = data.iloc[:,8].values

    models=[]

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, KFold

    models.append(('LR',LogisticRegression()))
    models.append(('Tree',DecisionTreeClassifier()))
    models.append(('SVC',SVC()))
    models.append(('NB',GaussianNB()))
    models.append(('knn',KNeighborsClassifier()))
    scoring = 'accuracy'
    print(models)

    results=[]
    names=[]

    for name,model in models:
        kfold = KFold(n_splits=10,random_state=10)
        a = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(a)
        names.append(name)
        print(name)
        print(a)
        print('==============================')

    fig=plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax=fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def bank():
    data = pd.read_csv('bank.csv')

    X = data.iloc[:,[0,1,3,5,6,10,11,12,13,15,16,17,18,19]].values
    y = data.iloc[:,-1].values
    print(X[5,:])
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder

    le = LabelEncoder()
    X[:,1]=le.fit_transform(X[:,1])
    X[:,2]=le.fit_transform(X[:,2])
    X[:,3]=le.fit_transform(X[:,3])
    X[:,4]=le.fit_transform(X[:,4])

    print(X[5,:])

    onehot = OneHotEncoder(categorical_features=[1,2,3,4])
    X = onehot.fit_transform(X).toarray()

    models=[]

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC 
    from sklearn.model_selection import cross_val_score, KFold

    models.append(('LR',LogisticRegression()))
    models.append(('Tree',DecisionTreeClassifier()))
    models.append(('SVC',SVC()))
    models.append(('NB',GaussianNB()))
    models.append(('knn',KNeighborsClassifier()))
    scoring = 'accuracy'
    print(models)

    results=[]
    names=[]

    for name,model in models:
        kfold = KFold(n_splits=10,random_state=10)
        a = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(a)
        names.append(name)
        print(name)
        print(a)
        print('==============================')

    fig=plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax=fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
 

def r71():
    from sklearn.datasets import load_iris 
    iris = load_iris()

    X = iris.data
    y = iris.target

    models=[]

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    models.append(('tree',DecisionTreeClassifier()))
    models.append(('LR',LogisticRegression()))
    models.append(('NB',GaussianNB()))
    models.append(('svc',SVC()))
    models.append(('knn',KNeighborsClassifier()))
    scoring='accuracy'
    results=[]
    names=[]
    from sklearn.model_selection import cross_val_score,KFold
    for name, model in models:
        kfold =KFold(n_splits=10,random_state=10)
        a=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(a)
        names.append(name)
        print(name)
        print(a)

    fig = plt.figure()
    fig.suptitle('Comparision algoriyhm')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def r73():
    data = pd.read_csv('bank.csv')
    print(data.isnull().sum())

    X = data.iloc[:,[0,1,3,5,6,10,11,12,13,15,16,17,18,19]].values
    y = data.iloc[:,-1].values 

    print(X[5,:])

    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    le = LabelEncoder()
    X[:,1]= le.fit_transform(X[:,1])
    X[:,2]= le.fit_transform(X[:,2])
    X[:,3]= le.fit_transform(X[:,3])
    X[:,4]= le.fit_transform(X[:,4])

    oh = OneHotEncoder(categorical_features=[1,2,3,4])
    X = oh.fit_transform(X).toarray()

    print(X[5,:])

    models=[]
    results=[]
    names=[]

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    models.append(('tree',DecisionTreeClassifier()))
    models.append(('LR',LogisticRegression()))
    models.append(('NB',GaussianNB()))
    models.append(('svc',SVC()))
    models.append(('knn',KNeighborsClassifier()))
    scoring='accuracy'
    from sklearn.model_selection import cross_val_score,KFold


    for name,model in models:
        kfold = KFold(n_splits=10,random_state=10)
        a = cross_val_score(model,X,y,cv=kfold,scoring=scoring)
        results.append(a)
        names.append(name)
        print(name)
        print(a)
    fig = plt.figure()
    fig.suptitle('Comparision')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(name)
    plt.show()

r73()