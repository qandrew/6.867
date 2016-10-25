# Oct 22 2016
# 6867 HW 2
# Andrew and Karan
import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code
from sklearn.linear_model import LogisticRegression

def load_data(set1, set2): #set1 is the +1 classified and set2 is the -1 classified
	X = []
	Y = []
	for digit in set1:
		temptrain = loadtxt('data/mnist_digit_'+str(digit)+'.csv') #5174 x 784
		for i in xrange(500): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			# add = np.matrix(temptrain[i][0:28]) #new dataset to add
			# for j in xrange(1,temptrain.shape[1]/28):
			# 	add = np.concatenate((add,np.matrix(temptrain[i][28*j:28*j+28])))
			add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X.append(add)
			Y.append(1)
		# for i in xrange(200,350): #validation size
		# 	add = np.matrix(temptrain[i][0:28]) #new dataset to add
		# 	for j in xrange(1,temptrain.shape[1]/28):
		# 		add = np.concatenate((add,np.matrix(temptrain[i][28*j:28*j+28])))
		# 	X_val.append(add)
		# 	Y_val.append(1)
		# for i in xrange(350,500): #test size
		# 	add = np.matrix(temptrain[i][0:28]) #new dataset to add
		# 	for j in xrange(1,temptrain.shape[1]/28):
		# 		add = np.concatenate((add,np.matrix(temptrain[i][28*j:28*j+28])))
		# 	X_test.append(add)
		# 	Y_test.append(1)
	for digit in set2:
		temptrain = loadtxt('data/mnist_digit_'+str(digit)+'.csv') #5174 x 784
		for i in xrange(500): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			# add = np.matrix(temptrain[i][0:28]) #new dataset to add
			# for j in xrange(1,temptrain.shape[1]/28):
			# 	add = np.concatenate((add,np.matrix(temptrain[i][28*j:28*j+28])))
			add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X.append(add)
			Y.append(-1)
	return np.matrix(X),Y

print '======Training======'
# load data from csv files
X,Y = load_data([1],[2])
print X.shape, len(Y)

# pl.imshow(X[600], cmap='Greys_r') #visualize number
# pl.show()

# Carry out training.
lr = LogisticRegression(penalty='l2', tol=0.0001, C=1000000.0, 
	fit_intercept=True, intercept_scaling=1, 
	solver='liblinear', max_iter=2000)
	#c is inverse of regularizer strength, large C means small regularizer
	#user liblinear for solver
	#specify in penalty to use l1 or l2

# # Define the predictLR(x) function, which uses trained parameters
lr.fit(X,Y)

def predictLR(x):
    return lr.predict_proba(x)[0,1]

plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')


