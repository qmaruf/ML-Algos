import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.datasets import make_blobs



n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=100)
plt.scatter(X[:, 0], X[:, 1])


k = 10
colors = []
import random
r = lambda: random.randint(0,255)
while len(set(colors))<k:
	colors.append('#%02X%02X%02X' % (r(),r(),r()))


centers = X[[i for i in range(k)]]
print centers

iterations = 100
plt.ion()

def get_nearest_center(point, centers):
	mindist = 1e99
	pos = -1
	for n, center in enumerate(centers):
		dist = distance.euclidean(point, center)
		if dist < mindist:
			mindist = dist
			pos = n	
	return pos

mp = dict()
import time
for it in range(iterations):
	print '%d/%d' % (it, iterations)
	for point in X:
		nearest_center_id = get_nearest_center(point, centers)		
		mp[(point[0], point[1])] = nearest_center_id

	for center_id in range(k):
		neighbors = []
		for key in mp:
			if mp[key] == center_id:
				neighbors.append(key)
		neighbors = np.array(neighbors)
		centers[center_id] = np.mean(neighbors, axis=0)
	if it%5 == 0:
		plt.clf()
		for point in X:
			plt.scatter(point[0], point[1], color=colors[mp[(point[0], point[1])]])
		plt.scatter(centers[:,0].astype(int), centers[:,1].astype(int), marker='x', s=150, color='#ff0000', linewidth=10)
		plt.draw()
		plt.pause(0.05)
			


print centers
print colors

