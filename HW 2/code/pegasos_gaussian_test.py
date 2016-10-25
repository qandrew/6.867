import numpy as np
from plotBoundary import *
import pylab as pl

# import your LR training code
def kernel_gaussian(a,b,gamma):
	power = np.linalg.norm(a - b)**2
	return e**(-gamma*power)
	# return np.dot(a,b.T)

### TODO: Compute the kernel matrix ###
def compute_ksums(X,gamma):
	n = X.shape[0]
	K = zeros((n,n));
	# K_sums = zeros((n,1)) # sum_j K(x_i, x_j)
	for i in xrange(n):
		for j in xrange(n):
			K[i,j] = kernel_gaussian(X[i],X[j],gamma)
			# K[i,j] = np.dot(np.array(X[i]),np.array(X[j]).T)
		# K_sums[i] = np.sum(K[i,:])
	return K

### TODO: Implement train_gaussianSVM ###
def train_gaussianSVM(X,Y,K,l,epochs):
	t = 0
	alpha = np.zeros((X.shape[0],1))	
	n = X.shape[0]
	for epoch in xrange(epochs):
		if epoch%100 == 0: print "epoch...", epoch
		for i in xrange(n):
			t += 1
			step = 1/(t*l)
			# if (Y[i]* alpha[i]*K_sums[i])[0] < 1:
			# if get_comparator(Y[i],alpha,K,i) < 1:
			if (Y[i]*np.dot(K[i,:].reshape(1,-1),alpha)) < 1:
				alpha[i] = (1 - step*l)*alpha[i] + step*Y[i]
			else:
				alpha[i] = (1 - step*l)*alpha[i]
	return alpha

# # Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
# ### TODO:  define predict_gaussianSVM(x) ###
def predict_gaussianSVM(x):
	toret = 0
	for i in xrange(n):
		# https://www.quora.com/What-is-the-intuition-behind-Gaussian-kernel-in-SVM
		toret += alpha[i]*kernel_gaussian(x,X[i],gamma)
	return toret

def predictSVM_gaussian(alpha,X,x,gamma): #for question 4
	toret = 0
	n = X.shape[0]
	for i in xrange(n):
		# https://www.quora.com/What-is-the-intuition-behind-Gaussian-kernel-in-SVM
		toret += alpha[i]*kernel_gaussian(x,X[i],gamma)
	return toret

if __name__ == "__main__":
	# load data from csv files
	name = '1'
	train = loadtxt('data/data'+name+'_train.csv')
	Xplot = train[:,0:2]
	X = np.matrix(train[:,0:2]) #formulate into NP matrix
	Y = train[:,2:3]
	n = X.shape[0]

	# Carry out training.
	epochs = 1000;
	lmbda = .02; #lambda
	gamma = 1;

	K = compute_ksums(X,gamma)

	alpha = train_gaussianSVM(X, Y, K, lmbda, epochs);
	print 'alpha computed' #alpha
	print np.linalg.norm(alpha)

	# # plot training results
	print '======PLOTTING======'
	plotDecisionBoundary(Xplot, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM with gamma='+str(gamma))
	# pl.savefig('pegasos_gaussian_data'+str(name)+'_gamma'+str(gamma)+'.png')
	pl.show()
	print 'done'