#Nov 8 2016

import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code
#from sklearn.linear_model import LogisticRegression

def regularize(w, loss, lamb):
	toret = loss(w)

def softmax(layer):
	top = np.exp(layer)
	bottom = np.sum(top)
	return np.divide(top, bottom)

def ReLU(z):
	return 0.5*(z + np.absolute(z))

def crossEntropy(y,layer):
	#this is the loss function
	softm = softmax(layer)
	softm = np.log(softm)
	toret = -np.dot(y,softm)
	return toret

def nnInitialize(XTrain,YTrain,layers):
	#initialize all weights of neural net to random weights distributed with N(0,1/sqrt(m))
	#layers is vector, determining width of each hidden layer
	n,d = XTrain.shape
	sqrtd = 1.0/np.sqrt(d)
	hiddenWeights = [] #array of matrices
	offsets = [] #array of offsets
	w1 = np.random.normal(0,sqrtd,[d,layers[0]]) #initialize
	b = np.random.normal(0,sqrtd,[layers[0],1])
	hiddenWeights.append(w1); offsets.append(b) #add them to the array
	for i in xrange(len(layers)-1):
		sqrtm = 1.0/np.sqrt(layers[i])
		w2 = np.random.normal(0,sqrtm,[layers[i],layers[i+1]])
		hiddenWeights.append(w2)
		b = np.random.normal(0,sqrtm, [layers[i+1],1])
		offsets.append(b)
	return hiddenWeights, offsets

def nnTrain(XTrain,YTrain,weights,offsets,maxIter=1000,debug=False):
	print 'initial accuracy', classifyAccuracy(XTrain,YTrain,weights,offsets)
	prevWeight = np.copy(weights)
	prevOffset = np.copy(offsets)

	for iter in xrange(maxIter):
		learningRate = 0.2/np.power(iter,1./3) #empirically tested
		index = np.random.randint(0,n) #for SGD
		x = XTrain[index].reshape(-1,1)
		y = YTrain[index]

		aggregated,activated = forwardProp(x,weights,offsets) #propagate weights forward through NN
		delta = backProp(y,weights,offsets,aggregated,activated)


def classifyAccuracy(X,Y,weights, offsets):
	#TO WORK
	#given X, Y values, weights and offsets for each neuron in the neural network, compute the error rate
	n1,d = X.shape
	n2.k = Y.shape
	assert n1 == n2
	correct = 0.0
	for i in xrange(len(X)):
		predict_y = nnPredict(X[i],weights,offsets)
		correct += np.dot(Y[i],predict_y)[0][0]
	accuracy = correct/n1
	return accuracy

def nnPredict(x,weights,offsets):
	#predict the class of x
	#TO WORK
	pass


def forwardProp(x, weights, offsets, debug = True):
	#x a dataset that we are feeding in
	prev_layer = x.reshape(-1,1)
	if debug: print prev_layer.shape
	aggregated = [] #keep track of aggregated vectors at each layer
	activated = []
	for i in xrange(len(weights)):
		W = weights[i]
		if debug: print i, W.T.shape
		layer_l = np.dot(W.T,prev_layer) + offsets[i]
		print i, 'input', layer_l.shape
		aggregated.append(layer_l)
		if i == len(weights)-1: #last layer
			result_l = softmax(layer_l)
			print i, 'softmax'
		else:
			result_l = ReLU(layer_l)
			print i, 'relu'
		print i, 'result', result_l.shape
		activated.append(result_l)
		prev_layer = result_l
	return aggregated,activated

def backProp(y,weights, offsets,aggregated,activated)

if __name__ == "__main__":
	# parameters
	print '======Training======'
	# load data from csv files
	train = loadtxt('data/data_3class.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	layers = [10,3]
	# print X.shape
	weights, offsets = nnInitialize(X,Y,layers)
	print weights[0].shape, weights[1].shape
	print 'done init'

	ag, ac = forwardProp(X[3], weights, offsets)
	# print ag[1]
	# print ac[1]
	print 'done foward prop'
	print len(ac), ac[0].shape, ac[1].shape










