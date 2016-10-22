import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = np.matrix(train[:,0:2]) #formulate into NP matrix
Y = train[:,2:3]
n = X.shape[0] #height of X

power = -2
l = 2**power #lambda value
C = 1/(l*n) #margin tolerance
epoch = 0
max_epochs = 1000
w = np.zeros((X[0].shape[1], 1))
w0 = 0 #the bias
t = 0

# Carry out training.
while epoch < max_epochs:
	for i in xrange(n):
		t += 1
		step = 1/(t*l)
		if Y[i]*(X[i]*w)[0,0] < 1: #if incorrectly classified, no bias
		# if Y[i]*(X[i]*w + w0)[0,0] < 1: #if incorrectly classified
			w = (1 - step*l)*w + (step*Y[i]*X[i]).T #w is 2x1 but x[i] is 1x2
			w0 = (1 - step*l)*w0 + (step*Y[i]).T #separate the bias term
		else: #correctly classified
			w = (1 - step*l)*w
			w0 = (1 - step*l)*w0
	epoch += 1

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###
print 'w', w
print 'w0', w0
print 'lambda power', power

def predict_linearSVM(x):
	return w[0]*x[0] + w[1]*x[1] # + w0 #w0 is the bias

# plot training results
print '======PLOTTING======'
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.title("lambda=2^" +str(power)) 
pl.show()
