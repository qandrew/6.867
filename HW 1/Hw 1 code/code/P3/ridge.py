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

# def ridge(X,Y,l,M):
#     X_o = X
#     X = vandermonde(X_o,M)
#     reg = l*np.identity(M+1)
#     theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
#
#     X_basis = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1
#     X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
#     plt.title('M=' + str(M))
#     Y_plotting = X_plotting.dot(theta)
#     x, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))
#     plt.legend(handles=[x])
#     plt.plot(X_o,Y,'co')
#     plt.xlabel('x')
#     plt.ylabel('y')
#
#     plt.show()
#
#     return theta

def ridge(X,Y,l_list,M):
    X_o = X
    M = 10
    X = vandermonde(X_o,M)
    X_basis = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1
    X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))

    plt.figure(1)
    plt.subplot(211)
    plt.title('M=10')

    l = 0
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    x, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = 1e-10
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    y, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = .1
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    z, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = 10
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    w, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    plt.legend(handles=[x,y,z,w])
    plt.plot(X_o,Y,'co')
    plt.xlabel('x')
    plt.ylabel('y')

    ####################################

    M = 6
    X = vandermonde(X_o,M)
    X_basis = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1
    X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))

    plt.subplot(212)
    plt.title('M=6')

    l = 0
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    x, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = 1e-5
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    y, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = .1
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    z, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    l = 10
    reg = l*np.identity(M+1)
    theta = np.linalg.inv(reg + np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
    Y_plotting = X_plotting.dot(theta)
    w, = plt.plot(X_basis,Y_plotting,label="lambda=" + str(l))

    plt.legend(handles=[x,y,z,w])
    plt.plot(X_o,Y,'co')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

    return theta

def vandermonde(X,M):
    #construct a vandermonde matrix with width M. input vars x
    X_new = np.ones((X.shape[0],M+1))
    for i in xrange(1,M+1):
        X_new[:,[i]] = np.matrix(np.power(X,i))
    return X_new

if __name__ == "__main__":

    X,Y = loadFittingDataP2.getData(False)
    X = np.transpose(np.matrix(X))
    Y = np.transpose(np.matrix(Y))
    L = 100
    M = 10

    ridge(X,Y,L,M)
