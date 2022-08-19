# -*- coding: utf-8 -*-
"""
Created on Tue Aug 9 12:10:48 2021

@author: Jagriti
"""

# LOADING DATASET
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data    #FEATURES
Y = iris.target  #TARGET


"""-----------------------------------------------------------------"""
# VISUALIZATION OF DATASET
import pandas as pd
df=pd.DataFrame(X)
df.columns= iris.feature_names
print(df)
print('target names' ,iris.target_names)

# print data set descriptions and class distributions
print (df.describe())


"""-----------------------------------------------------------------"""
# KMEANS CLUSTERING: ELBOW METHOD TO DETERMINE THE OPTIMAL NO. OF CLUSTERS
from sklearn.cluster import KMeans

# empty x and y data lists
x = []
y = []

for i in range(1,31):
    # initialize and fit the kmeans model 
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    
    # append number of clusters to x data list
    x.append(i)
    
    # append average within-cluster sum of squares to y data list
    awcss = kmeans.inertia_ / df.shape[0]
    y.append(awcss)
    
    
    
import matplotlib.pyplot as plt
plt.plot(x,y, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Within-Cluster Sum of Squares')
plt.title('K-Means Clustering Elbow Method')
plt.show()    
print('optimal number of clusters = the one present exactly on the elbow.') 

   
"""-----------------------------------------------------------------"""
#TO FIND CO-RELATION WITHIN FEATURES : helpful for PCA   
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize = (10,10))
plt.show()    


"""-----------------------------------------------------------------"""
"""principal component analysis (PCA) is a statistical procedure that uses 
an orthogonal transformation to convert a set of observations of possibly
correlated variables into a set of values of linearly uncorrelated variables
 called principal components.
 
 Its something like eigen vectors & eigen values"""

#PERFORMING PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=(2)) # 4D ----> 2D
pc = pca.fit_transform(df)

# re-fit kmeans model to principle components with optimal no of clusters
kmeans = KMeans(n_clusters = 3)
kmeans.fit(pc)



"""-----------------------------------------------------------------"""
# VIZUALIZATION

# set size for the mesh
h = 0.02 # determines quality of the mesh [x_min, x_max]x[y_min, y_max]

import numpy as np
# generate mesh grid
x_min, x_max = pc[:, 0].min() - 1, pc[:, 0].max() + 1
y_min, y_max = pc[:, 1].min() - 1, pc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# label each point in mesh using last trained model
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# generate color plot from results
Z = Z.reshape(xx.shape)
plt.figure(figsize = (10, 10))
plt.clf()
plt.imshow(Z, interpolation = 'nearest',
          extent = (xx.min(), xx.max(), yy.min(), yy.max()),
          cmap = plt.cm.tab20c,
          aspect = 'auto', origin = 'lower')

# plot the principle components on the color plot
for i, point in enumerate(pc):
    if Y[i] == 0:
        plt.plot(point[0], point[1], 'g.', markersize = 10)
    if Y[i] == 1:
        plt.plot(point[0], point[1], 'r.', markersize = 10)
    if Y[i] == 2:
        plt.plot(point[0], point[1], 'b.', markersize = 10)

# plot the cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 250,
            linewidth = 4, color = 'w', zorder = 10)


plt.title('K-Means Clustering on PCA-Reduced Iris Data Set')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks(())
plt.yticks(())
plt.show()    


"""-----------------------------------------------------------------"""
#MEARURING ACCURACY AFTER PCA

"""some common clustering metrics, such as homogeneity, completeness, and 
V-measure.

Homogeneity - measures whether or not all of its clusters contain only data
 points which are members of a single class.
 
Completeness - measures whether or not all members of a given class are 
elements of the same cluster

V-measure - the harmonic mean between homogeneity and completeness """

from sklearn import metrics
#K MEANS CLUSTERING ON NON REDUCED DATA
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(X)


#K MEANS CLUSTERING ONREDUCED DATA
kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(pc)

#CLUSTERING METRICS
print('For non reduced data')
print('Homogeneity =', metrics.homogeneity_score(Y, kmeans1.labels_))
print('Completeness =', metrics.completeness_score(Y, kmeans1.labels_))
print('V-measure =',metrics.v_measure_score(Y, kmeans1.labels_))

print('PCA reduced data')
print('Homogeneity =', metrics.homogeneity_score(Y, kmeans2.labels_))
print('Completeness =', metrics.completeness_score(Y, kmeans2.labels_))
print('V-measure =',metrics.v_measure_score(Y, kmeans2.labels_))





















