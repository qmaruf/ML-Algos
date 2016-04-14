__author__ = 'maruf'

# univariate linear regression
# batch gradient descent

import numpy as np
from sklearn.datasets import make_regression
import pylab

n_features = 1
n_samples = 1000
n_train = int(n_samples * 0.9)
n_test = int(n_samples * 0.1)

x, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=1, random_state=0, noise=35)

x_train, y_train = x[:n_train], y[:n_train]
x_train = np.insert(x_train, 0, 1, axis=1)

x_test, y_test = x[n_train:], y[n_train:]
tmp = x_test
x_test = np.insert(x_test, 0, 1, axis=1)

theta = np.zeros((1, n_features + 1))[0]

prev_cost = 1e99
alpha = 0.1


'''
m = # sample
n = # feature

x[m x n]
theta[n x 1]
y[m x 1]

loss = x*theta - y [m x 1]

we need #n gradients to update #n features
multiplying the loss matrix using i'th feature will give i'th gradient
we can get nx1 matrix of gradients using x.transpose().loss


'''
while True:
    h = np.dot(x_train, theta)
    loss = h - y_train
    cost = sum(loss ** 2)/(n_train*1.0)
    print cost
    if prev_cost - cost < 0.0001:break
    prev_cost = cost
    gradients = np.dot(x_train.transpose(), loss)/n_train/1.0
    theta = theta - alpha * gradients


print theta

y_predict = []
for sample in x_test:
    p = np.dot(sample, theta)
    y_predict.append(p)

pylab.plot(tmp, y_test, 'o')
pylab.plot(tmp, y_predict)
pylab.show()
