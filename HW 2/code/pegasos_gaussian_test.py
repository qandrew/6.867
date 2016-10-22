import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = np.matrix(train[:,0:2]) #formulate into NP matrix
Y = train[:,2:3]
n = X.shape[0]

# Carry out training.
epochs = 1000;
lmbda = .02; #lambda
gamma = 2e-2;

def kernel_gaussian(a,b,gamma):
	power = np.linalg.norm(a - b)**2
	return e**(-gamma*power)

K = zeros((n,n));
K_sums = zeros((n,1)) # sum_j K(x_i, x_j)
### TODO: Compute the kernel matrix ###
for i in xrange(n):
	for j in xrange(n):
		K[i,j] = kernel_gaussian(X[i],X[j],gamma)
	K_sums[i] = np.sum(K[i,:])

### TODO: Implement train_gaussianSVM ###
def train_gaussianSVM(X,Y,K_sums,l,epochs):
	t = 0
	alpha = np.zeros((X.shape[0],1))	
	for epoch in xrange(epochs):
		for i in xrange(n):
			t += 1
			step = 1/(t*l)
			if (Y[i]* alpha[i]*K_sums[i])[0] < 1:
				alpha[i] = (1 - step)*alpha[i] + step*Y[i]
			else:
				alpha[i] = (1 - step)*alpha[i]
	return alpha

alpha = train_gaussianSVM(X, Y, K_sums, lmbda, epochs);

print alpha

# # Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
# ### TODO:  define predict_gaussianSVM(x) ###
def predict_gaussianSVM(x):
	toret = 0
	for i in xrange(n):
		# https://www.quora.com/What-is-the-intuition-behind-Gaussian-kernel-in-SVM
		toret += alpha[i]*kernel_gaussian(x,X[i],gamma)*Y[i]
	return toret

# # plot training results
print '======PLOTTING======'
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()
