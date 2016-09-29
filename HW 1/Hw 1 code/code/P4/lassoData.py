import pdb
import random
import pylab as pl
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getTrue():
    data = pl.loadtxt('lasso_true_w.txt')
    w = data[0:1].T
    return w

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def lasso(X,Y,L,Xval,Yval,ifPlotData=True):
    X_o = X
    X = vandermonde(X_o)
    clf = linear_model.Lasso(alpha=L)
    clf.fit(X,Y)
    theta = clf.coef_

    # if ifPlotData:
    #     X_basis = np.matrix([np.linspace(-3,3,600)]).transpose() #100 evenly spaced between 0,1
    #     X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
    #     Y_plotting = X_plotting.dot(theta)
    #     plt.plot(X_basis,Y_plotting)
    #     plt.plot(X_o,Y,'co')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()

    return checkFit(X_o,Y,Xval,Yval,theta,L)

    # return checkFit(X_o,Y,targetX,targetY,theta,M,l,testX,testY)


def vandermonde(X):
    #construct a vandermonde matrix with width dim. input vars x
    X_new = np.ones((X.shape[0],13))
    X_new[:,[0]] = np.matrix(X)
    for i in xrange(1,13):
        X_new[:,[i]] = np.matrix(np.sin(0.4*math.pi*X*i)) #as specified in prob 4
    return X_new

def checkFit(Xin,Yin,targetX, targetY, theta,l):
    Xv = vandermonde(targetX)
    predictY = Xv.dot(theta)
    # Xt = vandermonde(testX,M)
    # predictY_test = Xt.dot(theta)

    sseV = squareError(predictY,targetY)
    # sseT = squareError(predictY_test,testY)

    X_basis = np.matrix([np.linspace(-1,1,200)]).transpose() #100 evenly spaced between 0,1
    X_plotting = vandermonde(X_basis) # np.ones((X_basis.shape[0],M))
    Y_plotting = X_plotting.dot(theta)
    # w, = plt.plot(testX,testY,'go',label="test data")
    x, = plt.plot(X_basis,Y_plotting,'black',label="fit curve")
    y, = plt.plot(targetX,targetY,'bo',label="validation data")
    z, = plt.plot(Xin,Yin,'ro',label="training data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("lambda=" + str(l) + ", Validation SSE=" + str(sseV))
    plt.legend([x,z,y],loc=4)
    plt.show()

    return

def squareError(current,target):
    sse = np.sum(np.square(target-current))
    return sse

if __name__ == "__main__":

    Xa,Ya = lassoTrainData()
    Xb,Yb = lassoTestData()
    Xv,Yv = lassoValData()
    w_true = getTrue()

    L=0

    lasso(Xa,Ya,L,Xv,Yv)
