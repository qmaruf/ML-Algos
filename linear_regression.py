import time
import numpy as np
from sklearn.datasets import make_regression
import pylab

n_features = 1
n_samples = 100
n_train = int(n_samples * 0.9)
n_test = int(n_samples * 0.1)

x, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=1, random_state=0, noise=35)

x_train, y_train = x[:n_train], y[:n_train]
x_train = np.insert(x_train, 0, 1, axis=1)

x_test, y_test = x[n_test:], y[n_test:]
tmp = x_test
x_test = np.insert(x_test, 0, 1, axis=1)

theta = np.zeros((1, n_features + 1))[0]


def calculate_cost(theta):
    cost = 0
    for i in range(n_train):
        cost += (np.dot(x_train[i], theta) - y_train[i]) ** 2
    return cost / 2.0 / n_train


prev_cost = 1e99
alpha = 0.1

while True:
    cost = calculate_cost(theta)
    if prev_cost - cost < 0.0001: break
    prev_cost = cost
    # print cost
    tmptheta = []
    for j in range(len(theta)):
        grad = 0
        for i in range(n_train):
            grad += (np.dot(theta, x_train[i, :]) - y_train[i]) * x_train[i][j]
        grad /= n_train * 1.0
        tmptheta.append(theta[j] - alpha * grad)
    theta = tmptheta


print theta

y_predict = []
for sample in x_test:
    p = np.dot(sample, theta)
    y_predict.append(p)

pylab.plot(tmp, y_test, 'o')
pylab.plot(tmp, y_predict)
pylab.show()