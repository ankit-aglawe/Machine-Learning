'''

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6)

kmeans= KMeans(n_clusters=4)
kmeans.fit(X)

y_means= kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], s=50, c=y_means, cmap="viridis")

plt.show()


#-----------------------------------------------------------------------



from sklearn.datasets.samples_generator import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X,y_true=make_moons(200,noise=0.05)
kmeans=KMeans(2)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)
plt.scatter(X[:,0],X[:,1],c=y_kmeans)
plt.show()

from sklearn.cluster import SpectralClustering
model=SpectralClustering(2,affinity='nearest_neighbors')

#model=SpectralClustering(2,affinity='nearest_neighbors',assign_labels='kmeans')
labels=model.fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.show()

#-------------------------------------------------------------------------


'''
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/home/ai1/My_Files/ML/Day4/s1.txt", delimiter='\s+')

print(data)
d1 = data['V1'].values
d2 = data['V2'].values

X = np.array(zip(d1,d2))


kmeans = KMeans(n_clusters=15)

kmeans.fit(X)

y_means = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1], s=50)

plt.show()

plt.scatter(X[:,0],X[:,1], s=60, c= y_means, cmap='viridis')

plt.show()
