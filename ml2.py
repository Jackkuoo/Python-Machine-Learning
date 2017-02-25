import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import linear_model
#from sklearn.neural_network import MLPClassifier
from sklearn import svm
boston = load_boston()
TrainX = boston.data[1:400,6].reshape(399,1)
TrainY = boston.target[1:400]
TestX = boston.data[401:,6].reshape(-1,1)
TestY = boston.target[401:]
'''
regr = linear_model.LinearRegression(15)
regr.fit(TrianX,TrianY);
'''
clf = svm.LinearSVR(degree=15,tol=1e-5)
clf.fit(TrainX,TrainY)

predictY = clf.predict(TestX)
print predictY 

fig ,ax = plt.subplots();
ax.scatter(TestX,TestY,color='blue')
ax.scatter(TestX,predictY,color='red')

plt.show()
