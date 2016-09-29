import pdb
import random
import pylab as pl
import numpy as np
import math
import loadFittingDataP2
import matplotlib.pyplot as plt

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

def ridge(X,Y,l,M,targetX,targetY,testX,testY,ifPlotData=True):
    X_o = X
    X = vandermonde(X_o,M)
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)

    # if ifPlotData:
    #     X_basis = np.matrix([np.linspace(-3,3,600)]).transpose() #100 evenly spaced between 0,1
    #     X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
    #     Y_plotting = X_plotting.dot(theta)
    #     plt.plot(X_basis,Y_plotting)
    #     plt.plot(X_o,Y,'co')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()

    print "Theta", theta

    return checkFit(X_o,Y,targetX,targetY,theta,M,l,testX,testY)


def vandermonde(X,M):
    #construct a vandermonde matrix with width M. input vars x
    X_new = np.ones((X.shape[0],M+1))
    for i in xrange(1,M+1):
        X_new[:,[i]] = np.matrix(np.power(X,i))
    return X_new

def checkFit(Xin,Yin,targetX, targetY, theta, M,l,testX,testY):
    Xv = vandermonde(targetX,M)
    predictY = Xv.dot(theta)
    Xt = vandermonde(testX,M)
    predictY_test = Xt.dot(theta)

    sseV = squareError(predictY,targetY)
    sseT = squareError(predictY_test,testY)

    X_basis = np.matrix([np.linspace(-3,3,600)]).transpose() #100 evenly spaced between 0,1
    X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
    Y_plotting = X_plotting.dot(theta)
    w, = plt.plot(testX,testY,'go',label="test data")
    x, = plt.plot(X_basis,Y_plotting,'black',label="fit curve")
    y, = plt.plot(targetX,targetY,'bo',label="validation data")
    z, = plt.plot(Xin,Yin,'ro',label="training data")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('M=' + str(M) + ", lambda=" + str(l) + ", Validation SSE=" + str(sseV)+ ", Test SSE=" + str(sseT))
    plt.legend(handles=[x,z,y,w],loc=4)
    plt.show()

    return

def squareError(current,target):
    sse = np.sum(np.square(target-current))
    return sse

if __name__ == "__main__":

    Xa,Ya = regressAData()
    Xb,Yb = regressBData()
    Xv,Yv = validateData()

    M=10
    L=1

    ridge(Xb,Yb,L,M,Xv,Yv,Xa,Ya)

    #x01 = [[ 0.87658229][ 0.90875408]]
    #x02 = [[ 0.99711996][ 0.88782698]-0.0374199 ]]
    #x04 =
    #x05 = [[ 0.60503682][ 1.1303877 ][ 0.30816594][-0.13840606][-0.04149159][ 0.01327405]]
