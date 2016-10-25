# Oct 22 2016
# 6867 HW 2
# Andrew and Karan
import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code
from sklearn.linear_model import LogisticRegression
import svm_test
import pegasos_linear_test as pegLin
import pegasos_gaussian_test as pegG

def load_data(set1, set2,normalize=True): #set1 is the +1 classified and set2 is the -1 classified
	X_train = []; X_val = []; X_test = []
	Y_train = []; Y_val = []; Y_test = []

	for digit in set1:
		temptrain = loadtxt('data/mnist_digit_'+str(digit)+'.csv') #5174 x 784
		for i in xrange(200): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_train.append(add)
			Y_train.append(1)
		for i in xrange(200,350): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_val.append(add)
			Y_val.append(1)
		for i in xrange(350,500): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_test.append(add)
			Y_test.append(1)
	for digit in set2:
		temptrain = loadtxt('data/mnist_digit_'+str(digit)+'.csv') #5174 x 784
		for i in xrange(200): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_train.append(add)
			Y_train.append(-1)
		for i in xrange(200,350): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_val.append(add)
			Y_val.append(-1)
		for i in xrange(350,500): #0-199 is train, 200-349 is val, 350-499 is test
			add = np.array(temptrain[i])
			if normalize: add = 2.0*add/255 - 1 #normalization
			add = add.tolist()
			X_test.append(add)
			Y_test.append(-1)
	# return np.matrix(X),Y
	return np.matrix(X_train),Y_train, np.matrix(X_val), Y_val, np.matrix(X_test), Y_test

def validate_lr(X_train, Y_train, X_val,Y_val):
	#get num correct, num incorrect
	print "\n===VALIDATING LOGISTIC REGRESSION==="
	lr = LogisticRegression(penalty='l2', tol=0.0001, C=1000000.0, 
	fit_intercept=True, intercept_scaling=1, 
	solver='liblinear', max_iter=2000)
	# # Define the predictLR(x) function, which uses trained parameters
	lr.fit(X_train,Y_train)

	correct = 0
	incorrect = []
	for i in xrange(len(X_val)):
		x = lr.predict_proba(X_val[i])
		if x[0,0] < x[0,1]: #predicting a +1
			if Y_val[i] == 1: correct += 1
			else: incorrect.append(i)
		else:
			if Y_val[i] == -1: correct += 1
			else: incorrect.append(i) 
	print "NUM CORRECT", correct
	print "NUM TOTAL", len(X_val)
	print "PERCENTAGE", correct*1.0/len(X_val)

	return incorrect

def validate_svm_lin(X_train, Y_train, X_val,Y_val):
	print "\n===VALIDATING SVM LINEAR KERNEL==="
	w = svm_test.linear_get_svm_ws_q4(X_train,np.matrix(Y_train).T, 1) #1x785 horiz array, w[0] is bias
	correct = 0
	incorrect = []
	for i in xrange(len(X_val)):
		x = svm_test.predictSVM_linear(w,X_val[i])
		if x > 0: #predicting a +1
			if Y_val[i] == 1: correct += 1
			else: incorrect.append(i)
		elif x < 0:
			if Y_val[i] == -1: correct += 1
			else: incorrect.append(i) 
		else: incorrect.append(i) 
	print "NUM CORRECT", correct
	print "NUM TOTAL", len(X_val)
	print "PERCENTAGE", correct*1.0/len(X_val)

	return incorrect

def validate_svm_gauss(X_train, Y_train, X_val,Y_val,c=1):
	"""TO BE FIXED"""
	print "\n===VALIDATING SVM GAUSS KERNEL==="
	b,alphas = svm_test.gaussian_get_svm_ws(X_train,np.matrix(Y_train).T,c)
	correct = 0
	incorrect = []
	for i in xrange(len(X_val)):
		x = svm_test.predictSVM_gauss(X_test,Y_test,b,alphas,X_val[0])
		if x > 0: #predicting a +1
			if Y_val[i] == 1: correct += 1
			else: incorrect.append(i)
		elif x < 0:
			if Y_val[i] == -1: correct += 1
			else: incorrect.append(i) 
		else: incorrect.append(i) 
	print "NUM CORRECT", correct
	print "NUM TOTAL", len(X_val)
	print "PERCENTAGE", correct*1.0/len(X_val)

	return incorrect

def validate_peg_lin(X_train, Y_train, X_val,Y_val,L=2e-5,max_epochs=1000):
	#get num correct, num incorrect
	print "\n===VALIDATING PEGASOS LINEAR==="
	w = pegLin.train_linearSVM(X_train,np.matrix(Y_train).T,L,max_epochs,show=False)
	correct = 0
	incorrect = []
	for i in xrange(len(X_val)):
		x = svm_test.predictSVM_linear(w,X_val[i])
		if x > 0: #predicting a +1
			if Y_val[i] == 1: correct += 1
			else: incorrect.append(i)
		elif x < 0:
			if Y_val[i] == -1: correct += 1
			else: incorrect.append(i) 
		else: incorrect.append(i) 
	print "NUM CORRECT", correct
	print "NUM TOTAL", len(X_val)
	print "PERCENTAGE", correct*1.0/len(X_val)

	return incorrect

	# w = pegLin.train_linearSVM(X_train,np.matrix(Y_train).T,L=2e-1,max_epochs=1000,show=False)
	# print w
	# print w.shape
	# print svm_test.predictSVM_linear(w,X_val[3])
	# print Y_val[3]

def validate_peg_gauss(X_train, Y_train, X_val,Y_val,gamma=1,L=2e-5,max_epochs=1000):
	#get num correct, num incorrect
	print "\n===VALIDATING PEGASOS GAUSS==="
	K = pegG.compute_ksums(X_train,gamma)
	alpha = pegG.train_gaussianSVM(X_train,Y_train,K,L, max_epochs)
	print "got norm(alpha) as", np.linalg.norm(alpha)
	correct = 0
	incorrect = []
	for i in xrange(len(X_val)):
		x = pegG.predictSVM_gaussian(alpha,X_train,X_val[30],gamma)
		if x > 0: #predicting a +1
			if Y_val[i] == 1: correct += 1
			else: incorrect.append(i)
		elif x < 0:
			if Y_val[i] == -1: correct += 1
			else: incorrect.append(i) 
		else: incorrect.append(i) 
	print "NUM CORRECT", correct
	print "NUM TOTAL", len(X_val)
	print "PERCENTAGE", correct*1.0/len(X_val)

	return incorrect

if __name__ == '__main__':
	
	print '======Training======'
	# load data from csv files
	X_train,Y_train,X_val,Y_val,X_test,Y_test = load_data([1],[3],normalize=True)

	# Carry out LR training.
	# lr_wrong = validate_lr(X_train,Y_train,X_val,Y_val)

	# Carry out linear SVM Training
	# svm_lin_wrong = validate_svm_lin(X_train,Y_train,X_val,Y_val)

	# Carry out Gaussian SVM Training
	# svm_gauss_wrong = validate_svm_gauss(X_train,Y_train,X_val,Y_val,c=0.001)

	# Carry out Pegasos Linear Training
	# pegLin_wrong = validate_peg_lin(X_train, Y_train, X_val,Y_val,L=2e-5,max_epochs=1000)
	# print pegLin_wrong

	# Carry out Pegasos Gaussian Training
	pegG_wrong = validate_peg_gauss(X_train, Y_train, X_val,Y_val,gamma=1,L=2e-5,max_epochs=1000)
	print pegG_wrong
	# K = pegG.compute_ksums(X_train,gamma=1)
	# alpha = pegG.train_gaussianSVM(X_train,Y_train,K,l = 0.02, epochs = 1000)
	# print np.linalg.norm(alpha)
	# print pegG.predictSVM_gaussian(alpha,X_train,X_val[30],gamma=1)



"""spare code"""
# def predictLR(x):
#     return lr.predict_proba(x)[0,1]

# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

# pl.imshow(X[600], cmap='Greys_r') #visualize number
# pl.show()

