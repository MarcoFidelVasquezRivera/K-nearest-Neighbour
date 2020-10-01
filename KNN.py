#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/MarcoFidelVasquezRivera/K-nearest-Neighbour/blob/master/KNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()
iris


# In[ ]:


X = iris.data[:, :2]
y = iris.target
plt.scatter(X[:,0],X[:,1])


# In[ ]:


import math

class KNN:

  def __init__(self,k):
    self.k = k

  def eucledian_distance(self,x,y):
    """    Returns the euclidean distance between x and y.

            Preconditions:
                    the dimensions of x and y must be the same

            Parameters:
                    x (n-dimensional numpy array): 
                    y (n-dimensional numpy array): Another decimal integer

            Returns:
                    binary_sum (int): Binary string of the sum of a and b.
    """
    eucledian_distance = np.sqrt(np.sum(np.square(x-y)))
    return eucledian_distance
    #distance = math.sqrt(sum()) para hacerlo despues
  
  def get_k_nearest_neighboors(self, point, data):
         """    Returns the K nearest neighboors of the point.

            Preconditions:
                    the dimensions of point and the points on data must be the same

            Parameters:
                    point (n-dimensional numpy array): an array of integers
                    data (an array of n-dimensional numpy arrays): an array of arrays of integers

            Returns:
                    binary_sum (int): Binary string of the sum of a and b.
    """
    distances = np.array([ self.eucledian_distance(data[point], datapoint) for datapoint in data ])
    sorted_distances = sorted(distances)
    indexes, = np.where(distances < sorted_distances[self.k])[:self.k]
    return indexes


# In[ ]:


knn = KNN(3)
knn.get_k_nearest_neighboors(2,X)
#knn.eucledian_distance(np.array([1,2,3]),np.array([5,6,7]))
#testdb = [2,4,7,14,1,8]
#knn.get_k_nearest_neighboors(5,testdb)
#X[1]


# In[ ]:


testdb_two = np.array([[2.0,3.4],[4.0,5.0],[3.0,8.0],[4.0,10.2],[20.0,0.0]])
point = [7.0,11.0]
for i in range(5):
  print(knn.eucledian_distance(point,testdb_two[i]))

knn.get_k_nearest_neighboors(point,testdb_two)

