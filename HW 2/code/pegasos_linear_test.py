import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
Xplot = train[:,0:2]
X = np.matrix(train[:,0:2]) #formulate into NP matrix
n = X.shape[0] #height of X
Xwide = np.hstack((np.ones((n,1)),X))
Y = train[:,2:3]

power = -10
l = 2**power #lambda value
C = 1.0/(l*n) #margin tolerance
epoch = 0
max_epochs = 1000
w = np.zeros((1, Xwide.shape[1])) #w[0] is the bias
t = 0

# Carry out training.
while epoch < max_epochs:
	if epoch%100 == 0: print "Epoch", epoch, w
	for i in xrange(n):
		t += 1
		step = 1.0/(t*l)
		w_unbiased = np.copy(w)
		w_unbiased[0][0] = 0
		if Y[i]*np.dot(w,Xwide[i].T) < 1: #incorrect classify
			w = w- step*l*w_unbiased + step*Y[i]*Xwide[i]
		else: #correctly classified
			w = w- step*l*w_unbiased
	epoch += 1

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###
print 'w', w
print 'lambda power', power

def predict_linearSVM(x):
	x = np.concatenate(([1],x))
	return np.dot(w,x.T)[0][0]

# plot training results
print '======PLOTTING======'
plotDecisionBoundary(Xplot, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM lambda = 2^'+str(power))
pl.show()
