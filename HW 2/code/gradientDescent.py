import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# Sept 20 2016
# Andrew Xia & Karan Kashyap
# 6.867 Machine Learning HW 1

def cost(X,y,theta,theta_0,L, debug = False):
    #compute J (cost) given theta
    reg = L * np.sum(np.square(theta))
    # theta_0 = np.ones((X.shape[0],1))*theta_0
    # if debug:
    #     print "\ndebug\n", np.exp(y[i]*(theta.T*X[i]+theta_0))
    tempCost = 0
    for i in xrange(len(X)): #slow
        tempCost += np.log(1 + np.exp(-y[i]*(X[i]*theta+theta_0)) )
    # cost = np.sum(np.log(1 + np.exp(-1*y.T*(X*theta + theta_0)))) + reg
    return tempCost + reg

# def updateThetaB(X,y,theta_old,iter,t,k):
    # #update theta
    # theta_new = np.zeros((X.shape[1],1)) #initiate theta at 0 #initiate theta_new to be 0s for now
    # step = (t+iter)**(-k)
    # # step = 1
    # for j in xrange(10):
    #     sum_error = np.sum((np.transpose(X[:,j])*(y-X*theta_old)))
    #     theta_new[j] = theta_old[j] + step*sum_error
    # return theta_new, cost(X,y,theta_new)

def updateThetaS(X,y,theta_old,theta_0_old,i,t,k,L):
    #update theta
    theta_new = np.zeros((X.shape[1],1)) #initiate theta at 0 #initiate theta_new to be 0s for now
    step = (t+i)**(-k)
    i = randint(0,X.shape[1]-1)
    for j in xrange(len(theta_old)):
        theta_new[j] = theta_old[j] + step*y[i,0]*X[i,j]/(1+np.exp(y[i,0]*(X[i]*theta_old+theta_0_old))) + step*L*theta_old[j]
    theta_0_new = theta_0_old + step*y[i,0]*(1+np.exp(y[i,0]*(X[i]*theta_old+theta_0_old)))
    return theta_new, theta_0_new, cost(X,y,theta_new,theta_0_new,L)

# def gradientDescentBatch(X,y, theta, t,k, conv):
    # #X: given X data
    # #y: given y data
    # #step: step size
    # #conv: convergence criterion
    # #goal: solve theta
    # theta_new = theta
    # cost_new = cost(X,y,theta_new)
    # # print "starting cost", cost_new
    # cost_old = cost_new + 2*conv
    # i = 0 #iterator counts
    # while abs(cost_old - cost_new) >= conv: #while we have not converged
    #     cost_old = cost_new
    #     theta_old = theta_new
    #     theta_new, cost_new = updateThetaB(X,y,theta_old,i,t,k)
    #     i += 1
    #     print "iteration:", i, "cost:", cost_new, "diff:", cost_old - cost_new
    #     # print "theta", theta_new

    # return theta_new

def gradientDescentStochastic(X,y, theta, theta_0, t, k, conv, L):
    #X: given X data
    #y: given y data
    #step: step size
    #conv: convergence criterion
    #goal: solve theta
    theta_0s = []
    theta_1s = []
    theta_2s = []
    theta_new = theta
    theta_new_0 = theta_0

    theta_0s.append(theta_new_0)
    theta_1s.append(theta_new[0,0])
    theta_2s.append(theta_new[1,0])

    cost_new = cost(X,y,theta_new,theta_new_0,L)
    # print "starting cost", cost_new
    cost_old = cost_new + 2*conv
    i = 0 #iterator counts
    while abs(cost_old - cost_new) >= conv: #while we have not converged
        cost_old = cost_new
        theta_old = theta_new
        theta_old_0 = theta_new_0
        theta_new, theta_new_0, cost_new = updateThetaS(X,y,theta_old,theta_old_0,i,t,k,L)
        i += 1
        #print "iteration:", i, "cost:", cost_new, "diff:", cost_old - cost_new, "weights:", theta_new, theta_new_0
        if i%200 == 0:
            print "iteration:", i, "cost:", cost_new, "weights:", theta_new, theta_new_0
        # print "theta", theta_new
        theta_0s.append(theta_new_0)
        theta_1s.append(theta_new[0,0])
        theta_2s.append(theta_new[1,0])

    return theta_new, theta_new_0, theta_0s, theta_1s, theta_2s

if __name__ == "__main__":

    name = '1'
    train = np.loadtxt('data/data'+name+'_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]
    X = np.matrix(X)
    Y = np.matrix(Y)

    theta = np.zeros((X.shape[1],1)) #initiate theta at 0, last element is w0
    theta_0 = 0
    step = 0.0002 #specify step here
    t = 100000 #specify variable step size
    k = 0.7
    conv = 0.000001 #specify convergence here

    # print cost(X,Y,np.matrix([[-1.21588605],[11.73331597]]),6.15088691,0)

    ### Lambda = 0 ###
    L = 0
    theta_f,theta_f_0, theta_0s, theta_1s, theta_2s = gradientDescentStochastic(X,Y,theta,theta_0,t,k,conv,L)

    print X.shape
    print Y.shape
    print X[0], Y[0]
    print len(theta_f), len(theta_f[0])
    print theta_f.shape, theta_f_0.shape

    print "initial cost", cost(X,Y,theta,theta_0,L)
    print "final theta & cost", theta_f, theta_f_0, cost(X,Y,theta_f,theta_f_0,L,debug=True)

    x, = plt.plot(theta_0s,label="w0")
    y, = plt.plot(theta_1s,label="w1")
    z, = plt.plot(theta_2s,label="w2")
    plt.xlabel('iterations')
    plt.ylabel('weight value')
    plt.title("lambda=" + str(0))
    # plt.legend(handles=[x,y,z],loc=2)
    plt.show()

    ### Lambda = 1 ###
    # L = 1
    # theta_f,theta_f_0, theta_0s, theta_1s, theta_2s = gradientDescentStochastic(X,Y,theta,theta_0,t,k,conv,L)
    # print "initial cost", cost(X,Y,theta,theta_0,L)
    # print "final theta & cost", theta_f, theta_f_0, cost(X,Y,theta_f,theta_f_0,L)

    # x, = plt.plot(theta_0s,label="w0")
    # y, = plt.plot(theta_1s,label="w1")
    # z, = plt.plot(theta_2s,label="w2")
    # plt.xlabel('iterations')
    # plt.ylabel('weight value')
    # plt.title("lambda=" + str(1))
    # # plt.legend(handles=[x,y,z],loc=2)
    # plt.show()
