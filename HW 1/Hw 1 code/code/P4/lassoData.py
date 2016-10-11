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
    return data

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

def lasso(X,Y,L,Xval,Yval,Xtest,Ytest,Wtrue,ifPlotData=True):
    X_o = X
    X = vandermonde(X_o)
    clf = linear_model.Lasso(alpha=L)
    clf.fit(X,Y)
    theta = np.transpose(np.matrix(clf.coef_))

    # if ifPlotData:
    #     X_basis = np.matrix([np.linspace(-3,3,600)]).transpose() #100 evenly spaced between 0,1
    #     X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
    #     Y_plotting = X_plotting.dot(theta)
    #     plt.plot(X_basis,Y_plotting)
    #     plt.plot(X_o,Y,'co')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()

    return checkFit(X_o,Y,Xval,Yval,theta,L,Xtest,Ytest,Wtrue)

    # return checkFit(X_o,Y,targetX,targetY,theta,M,l,testX,testY)


def vandermonde(X):
    #construct a vandermonde matrix with width dim. input vars x
    X_new = np.ones((X.shape[0],13))
    X_new[:,[0]] = np.matrix(X)
    for i in xrange(1,13):
        X_new[:,[i]] = np.matrix(np.sin(0.4*math.pi*X*i)) #as specified in prob 4
    return X_new

def checkFit(Xin,Yin,targetX, targetY, theta, l, testX, testY, trueW):
    Xv = vandermonde(targetX)
    predictY = Xv.dot(theta)
    Xt = vandermonde(testX)
    predictY_test = Xt.dot(theta)

    X_o = Xin
    X0 = vandermonde(X_o)
    clf0 = linear_model.Lasso(alpha=0)
    clf0.fit(X0,Yin)
    theta0 = np.transpose(np.matrix(clf0.coef_))

    reg = 2*np.identity(13)
    thetaR = np.linalg.inv(reg + np.transpose(X0).dot(X0)).dot(np.transpose(X0)).dot(Yin)

    sseV = squareError(predictY,targetY)
    sseT = squareError(predictY_test,testY)

    X_basis = np.matrix([np.linspace(-1,1,200)]).transpose() #100 evenly spaced between 0,1
    X_plotting = vandermonde(X_basis) # np.ones((X_basis.shape[0],M))
    Y_plotting = X_plotting.dot(theta)
    Y_true = X_plotting.dot(trueW)
    Y_0 = X_plotting.dot(theta0)
    Y_R = X_plotting.dot(thetaR)
    w, = plt.plot(testX,testY,'co',label="test data")
    x, = plt.plot(X_basis,Y_plotting,label="LASSO")
    y, = plt.plot(targetX,targetY,'bo',label="validation data")
    z, = plt.plot(Xin,Yin,'ro',label="training data")
    t, = plt.plot(X_basis,Y_true,label="true function")
    o, = plt.plot(X_basis,Y_0,label="lambda 0")
    r, = plt.plot(X_basis,Y_R,label="ridge")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("lambda=" + str(l) + ", Validation SSE=" + str(sseV) + ", Test SSE=" + str(sseT))
    plt.legend([x,z,y,w,t,o,r],loc=2)
    plt.show()

    print "W true", trueW #true weights
    print "LASSO", theta #for L=0.008
    print "ridge", thetaR #ridge
    print "Lambda 0", theta0 #setting lamba = 0

    return trueW, theta, thetaR, theta0

def squareError(current,target):
    sse = np.sum(np.square(target-current))
    return sse

def barPlot(theta, title):
	x = range(len(theta))
	print x
	width = 1/1.5
	plt.bar(x,theta,width,color='blue')
	plt.title(title)
	plt.xticks(x)
	plt.xlim(x[0],x[-1]+width) #force zeros
	plt.show()

if __name__ == "__main__":

    Xa,Ya = lassoTrainData()
    Xb,Yb = lassoTestData()
    Xv,Yv = lassoValData()
    w_true = np.transpose(np.matrix(getTrue()))

    L=0.008 #optimal

    trueW, theta, thetaR, theta0 = lasso(Xa,Ya,L,Xv,Yv,Xb,Yb,w_true)

    # barPlot(trueW,'trueW')
    # barPlot(theta,'estimated w with LASSO')
    # barPlot(thetaR,'estimated w with ridge')
    barPlot(theta0,'estimated w with lambda = 0')
