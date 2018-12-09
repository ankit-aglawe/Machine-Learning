from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split

data = load_digits()

X = data.images
y = data.target



lab = list(zip(X,y))

for index,(image,label) in enumerate(lab[:10]):
    plt.subplot(2,5,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(label)

plt.show()

n = len(X)

X1 = X.reshape(n, -1)

'''
X_train = X1[:n//2]
y_train = y[:n//2]

X_test = X1[n//2:]
y_test = y[n//2:]

'''

X_train, X_test, y_train, y_test=train_test_split(X1,y, test_size=0.5)

model = SVC(gamma =0.001)
model.fit(X_train, y_train)
p = model.predict(X_test)

#print(model.predict([[3,5,4,2]]))

print(p)

c=classification_report(y_test,p)

print(confusion_matrix(y_test,p))
print(c)


lab=list(zip(X[n//2:],p))


for index,(image,label) in enumerate(lab[:10]):
        plt.subplot(2,5,index+1)
        plt.axis('off')
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title(label)
plt.show()


