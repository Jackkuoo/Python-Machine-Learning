import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from numpy import genfromtxt
data = genfromtxt('X_train.csv', delimiter=',')

TrainX = data[1:30000]
TrainY = target[1:30000]

TestX = data[30001:]
TestY = target[30001:]

reg = linear_model.BayesianRidge()
reg.fit(TrainX, TrainY)

plt.scatter(TestX, TestY,  color='black')
plt.plot(TestX, reg.predict(TestX), color='blue',
         linewidth=3)


plt.show()
