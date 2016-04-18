from sklearn import datasets
import numpy as np
import math

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

def get_sigmoid(theta, X):
	return 1.0/(1 + np.e**(-X.dot(theta)))	
	
def get_cost(theta, X, y):		
	sigmoid_x = get_sigmoid(theta, X)
	cost_mat = -y*np.log(sigmoid_x)-(1-y)*np.log(1-sigmoid_x)
	cost = np.mean(cost_mat)
	return np.mean(cost)

def get_gradient(theta, X, y):
	sigmoid_x = get_sigmoid(theta, X)
	loss = sigmoid_x - y
	gradient = X.T.dot(loss)
	return gradient
	
def fit(X, y, alpha=0.001, iterations=1000):	
	theta = np.ones(X.shape[1]).reshape(-1, 1)	
	y = y.reshape(-1,1)
	for its in range(iterations):
		cost = get_cost(theta, X, y)
		gradient = get_gradient(theta, X, y)
		theta = theta - alpha*gradient
		print cost		
	return theta

n_train = int(len(X)*0.9)
theta = fit(X[:n_train], y[:n_train])
for x_test, y_test in zip(X[n_train:], y[n_train:]):
	pred = 1.0/(1+np.e**(-theta.T.dot(x_test)))
	print 'Original: %f Prediction %f' % (y_test, pred)
