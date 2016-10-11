import loadFittingDataP2
import math
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# Sept 24 2016
# Andrew Xia & Karan Kashyap
# 6.867 Machine Learning HW 1

def plotM(X,Y):
    t1,q = maxLikelihood(X,Y,1,ifPlotData=False)
    t0,q = maxLikelihood(X,Y,0,ifPlotData=False)
    t3,q= maxLikelihood(X,Y,3,ifPlotData=False)
    t10,q = maxLikelihood(X,Y,10,ifPlotData=False)

    X_basis = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1

    X_1 = vandermonde(X_basis,1)
    Y_1 = X_1.dot(t1)
    a1 = plt.plot(X_basis,Y_1, label="M=" + str(1))

    X_0 = vandermonde(X_basis,0)
    Y_0 = X_0.dot(t0)
    a0 = plt.plot(X_basis,Y_0, label="M=" + str(0))

    X_3 = vandermonde(X_basis,3)
    Y_3 = X_3.dot(t3)
    a3 = plt.plot(X_basis,Y_3,label="M=" + str(3))

    X_10 = vandermonde(X_basis,10)
    Y_10 = X_10.dot(t10)
    a10 = plt.plot(X_basis,Y_10,label="M=" + str(10))

    plt.plot(X,Y,'co')
    plt.legend([a0[0],a1[0],a3[0],a10[0]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return t0, t1, t3, t10

def maxLikelihood(X,Y,M,ifPlotData=True):
    #x is a 1D input vector
    #y is the output 
    #m is the max order of a simple polynomial basis
    #https://en.wikipedia.org/wiki/Polynomial_regression#Matrix_form_and_calculation_of_estimates
    X_new = vandermonde(X,M)
    theta = np.linalg.inv(np.transpose(X_new).dot(X_new)).dot(X_new.transpose()).dot(Y) # theta = (X^T X)^-1 X^T y

    X_cos = vandermonde(X,M,'cosine')
    thetaCos = np.linalg.inv(np.transpose(X_cos).dot(X_cos)).dot(X_cos.transpose()).dot(Y)

    if ifPlotData:
        X_basis = np.matrix([np.linspace(0,1,100)]).transpose() #100 evenly spaced between 0,1
        X_plotting = vandermonde(X_basis, M) # np.ones((X_basis.shape[0],M))
        Y_plotting = X_plotting.dot(theta)
        a, = plt.plot(X_basis,Y_plotting, label="polynomial M=" + str(M))
        X_pl_cos = vandermonde(X_basis,M,'cosine') #for 2-4 plotting
        Y_pl_cos = X_pl_cos.dot(thetaCos)
        b, = plt.plot(X_basis,Y_pl_cos, label="cosine M=" + str(M))
        plt.plot(X,Y,'co')
        plt.legend([a,b])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return theta, thetaCos

def vandermonde(X,dim,function='polynomial'):
    #construct a vandermonde matrix with width dim. input vars x
    X_new = np.ones((X.shape[0],dim+1))
    for i in xrange(1,dim+1):
        if function=='polynomial':
            X_new[:,[i]] = np.matrix(np.power(X,i))
        elif function=='cosine':
            X_new[:,[i]] = np.matrix(np.cos(i*math.pi*X)) #as specified in prob 2-4
    return X_new

def calculateResidualSquares(X,Y,theta):
    print X.shape
    print theta.shape
    X_new = vandermonde(X,theta.shape[0])
    new = Y - X_new.dot(theta)
    residual_squares =  np.sum(np.power(new,2)) #calculate RSS 
    deriv = thetaGradient(X_new,Y,theta)

    return residual_squares, deriv

def thetaGradient(X_new,Y,theta):
    #calculate the gradient of theta
    return 2*X_new.transpose().dot(X_new.dot(theta) - Y)    

def thetaLoss(X_new,Y,theta):
    #calculate the loss function at current theta
    return np.linalg.norm(X_new.dot(theta) - Y)

def gradientDescent(X,Y, M,step,conv,function='polynomial',theta =None):
    #running batch gradient descent on SSE Function
    if theta == None:
        #theta is our guess
        theta = np.zeros((M+1,1)) #start off with 0
    X_new = vandermonde(X,M,function) #create vandermonde for X
    diff = thetaLoss(X_new,Y,theta) #difference between current loss and previous loss
    prev = thetaLoss(X_new,Y,theta) #value of previous loss
    i = 0
    while abs(diff) > conv: # and i < 10:
        i += 1
        temp = thetaGradient(X_new,Y,theta)
        print 'grad', temp
        theta = theta - step*temp
        diff = thetaLoss(X_new,Y,theta) - prev
        prev = thetaLoss(X_new,Y,theta)
        print "iteration", i
        print "theta", theta
        print "diff", diff," conv", conv

    return theta


if __name__ == "__main__":
    (X,Y) = loadFittingDataP2.getData(ifPlotData=False)
    X = np.transpose(np.matrix(X))
    Y = np.transpose(np.matrix(Y))
    
    print X.shape
    print Y.shape
    
    M = 8 #choose M to your liking here.
    step = 0.1 #a step size too large may not work!
    conv = 0.01
    
    thetas = plotM(X,Y)

    for t in thetas:
        print t

    # theta, thetaCos = maxLikelihood(X,Y,M, ifPlotData=True)
    # print theta
    # print thetaCos

    # residual_squares, deriv = calculateResidualSquares(X,Y,theta)
    # print residual_squares
    # print deriv

    # theta = gradientDescent(X,Y,M,step,conv, function = 'polynomial')
    # print theta 