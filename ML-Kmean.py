
# coding: utf-8

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data


# In[4]:

print(X)


# In[6]:

resutl = KMeans(n_clusters=5, init='random')


# In[14]:

resutl.fit(X)


# In[15]:

resutl.labels_


# In[17]:

from sklearn.datasets import make_friedman3


# In[26]:

X, y = make_friedman3(n_samples=100, noise=0.0, random_state=0)
#print(X)
#print(y)
results = KMeans(n_clusters=5, init='random')
results.fit(X)


# In[27]:

results.labels_


# In[ ]:



