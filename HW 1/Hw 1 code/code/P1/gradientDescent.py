import loadFittingDataP1, loadParametersP1
import math
import numpy as np

# Sept 20 2016
# Andrew Xia & Karan Kashyap
# 6.867 Machine Learning HW 1

def gradientDescent(f, f_p, guess, step, conv):
    #f: function that we are given
    #f_p: (f' prime) optional function parameter test this function for f being the quadratic bowl and the negative Gaussian function
    #guess: initial guess
    #step: step size
    #conv: convergence criterion

    x_new = guess
    y_new = f(x_new)
    y_old = y_new + 2*conv
    i = 0 #iterator counts
    while abs(y_old - y_new) >= conv: #while we have not converged
        x_old = x_new
        y_old = y_new
        x_new = x_old - step*f_p(x_old) 
        y_new = f(x_new)
        i += 1
        diff = y_old - y_new
        print "iteration", i
        print "x_i", x_new
        print "y_i", y_new
        print "diff", diff," conv", conv

    return x_new, y_new

def finiteDifference(f,h,x):
    #problem 2 of gradient descent
    #f is the function
    #h is our step size, which is a scalar
    #x is where we are evaluation
    return f(x + 0.5*h) - f(x - 0.5*h)


def quadraticBowl(x):
    #A and b are global variables <- should change
    return float(0.5*np.transpose(x)*quadBowlA*x - np.transpose(x)*quadBowlb)

def quadraticBowlGrad(x):
    return quadBowlA*x - quadBowlb

def negGaussian(x):
    return -1/math.sqrt((2*math.pi)**3*np.linalg.det(cov))*np.exp(-0.5*np.transpose(x-mu)*np.linalg.inv(cov)*(x-mu))

def negGaussianGrad(x):
    # return -1*negGaussian(x)*np.linalg.inv(cov)*(x-mu)
    print 'neg gaus'
    print x
    toret = np.linalg.inv(cov)*(x-mu)*-1*negGaussian(x)
    print negGaussian(x)
    print ''
    return toret

if __name__ == "__main__":
    print "running gradientDescent.py"

    global quadBowlA
    global quadBowlb
    global mu
    global cov

    (mu,cov,quadBowlA,quadBowlb) = loadParametersP1.getData()
    mu = np.transpose(np.matrix(mu))
    quadBowlb = np.transpose(np.matrix(quadBowlb))
    #data =  mean, Gaussian covariance, A and b for quadratic bowl in order

    guess = np.transpose(np.matrix([[13,-7.3]])) #specify guess here
    step = 0.05 #specify step here
    conv = 0.0001 #specify convergence here

    print "mean", mu
    print "gauss cov", cov
    print "Quadratic Bowl A", quadBowlA
    print "Quadratic Bowl B", quadBowlb

    print "\n\n"

    x,y = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, step, conv)
    # x,y = gradientDescent(negGaussian, negGaussianGrad, guess, step, conv)
    print 'x',x
    print 'y',y
