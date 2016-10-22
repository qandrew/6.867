import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

debug = False

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = np.matrix(train[:,0:2]) #formulate into NP matrix
Y = train[:,2:3]
n = X.shape[0] #height of X

l = 0.1 #lambda value
C = 1/(l*n) #margin tolerance
epoch = 0
max_epochs = 1000
w = np.zeros((X[0].shape[1], 1))
t = 0

print X.shape, Y.shape

# print X[0].shape, w.shape
# print Y[0]*(X[0]*w) < 1

# Carry out training.
while epoch < max_epochs:
	for i in xrange(n):
		t += 1
		step = 1/(t*l)
		# print (w.T*X[i])
		if Y[i]*(X[i]*w)[0,0] < 1: #incorrectly classified
			w = (1 - step*l)*w + (step*Y[i]*X[i]).T #w is 2x1 but x[i] is 1x2
			if debug:
				print (step*Y[i]*X[i]).T
				print i, 'true'
				print w
		else: #correctly classified
			w = (1 - step*l)*w
			if debug:
				print i, 'false'
				print w
	epoch += 1

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###
print w

global count
count = 0
def predict_linearSVM(x):
	return w[0]*x[0] + w[1]*x[1]

# plot training results
print '======PLOTTING======'
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
print 'count', count
pl.show()
