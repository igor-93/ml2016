import numpy as np
from operator import itemgetter


class MyFeatSelector:
	indices = []
	k = 0

	def __init__(self, k):
		self.k = k

	def fit(self, X,y):
		h_means = []
		i_means = []
		#for it, x in enumerate(X):
		#	if y[it] == 0:
		#		i_means.append(x) 
		#	if y[it] == 1:
		#		h_means.append(x)
		
		ave = np.mean(X, axis=0)
  
		#i_means = np.mean(i_means, axis=0)
		#h_means = np.mean(h_means, axis=0)
		#diff = np.abs(np.subtract(h_means, i_means))

		#self.indices = diff.argsort()[-self.k:][::-1]
		#self.indices.sort()
		self.indices = [it for it, item in enumerate(ave) if item > self.k]
		#print 'indices', self.indices
		return self

	def transform(self, X):
		res = np.zeros((X.shape[0], len(self.indices)))
		for it, x in enumerate(X):
			res[it] = x[self.indices]

		return res

	def get_params(self, deep):
		return {'k':self.k}	

	def __str__(self):
		return 'k = ', self.k

		