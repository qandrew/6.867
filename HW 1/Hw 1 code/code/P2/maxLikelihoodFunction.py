import loadFittingDataP2
import math
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# Sept 24 2016
# Andrew Xia & Karan Kashyap
# 6.867 Machine Learning HW 1

def maxLikelihood(X,Y,M, ifPlotData=True):
    #x is a 1D input vector
    #y is the output 
    #m is the max order of a simple polynomial basis
    #https://en.wikipedia.org/wiki/Polynomial_regression#Matrix_form_and_calculation_of_estimates
    X_new = np.ones((X.shape[0],M))
    print X_new.shape
    print X_new
    for i in xrange(1,M):
        # print "ITER", i
        # print np.power(X,i).shape #creating new X matrix
        # print X_new[:,[i]].shape
        X_new[:,[i]] = np.matrix(np.power(X,i))
    maxVector = np.linalg.inv(np.transpose(X_new).dot(X_new)).dot(X_new.transpose()).dot(Y) # theta = (X^T X)^-1 X^T y

    if ifPlotData:
        # X_plotting = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1
        # print X_plotting
        Y_plotting = X_new.dot(maxVector)
        # Y_new = X_new.dot(maxVector)
        # print Y_new
        plt.plot(X,Y_plotting)
        plt.plot(X,Y,'co')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return maxVector

if __name__ == "__main__":
    (X,Y) = loadFittingDataP2.getData(ifPlotData=False)
    X = np.transpose(np.matrix(X))
    Y = np.transpose(np.matrix(Y))
    
    print X.shape
    print Y.shape
    
    M = 3 #choose M to your liking here.
    
    maxVector = maxLikelihood(X,Y,M)
    
    print maxVector


    