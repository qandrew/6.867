import numpy as np
from plotBoundary import *
import pylab as pl
# import your SVM training code
from cvxopt import matrix,solvers

# X = np.matrix([[2,2],[2,3],[0,-1],[-3,-2]])
# Y = np.array([[1.0],[1.0],[-1.0],[-1.0]])
# P = np.zeros((len(X),len(X)))
# for i in range(len(X)):
#     for j in range(len(X)):
#         P[i,j] = Y[i]*Y[j]*(X[i]*X[j].T)[0,0]
# P = matrix(P,tc='d')
# print P
# q = matrix([-1.0,-1.0,-1.0,-1.0])
# print q
# G = matrix(np.matrix([[-1.0,0.0,0.0,0.0],[0.0,-1.0,0.0,0.0],[0.0,0.0,-1.0,0.0],[0.0,0.0,0.0,-1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]),tc='d')
# print G
# h = matrix([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0])
# A = matrix([[Y[i]] for i in range(len(X))],tc='d')
# b = matrix([0.0])
# sol = solvers.qp(P,q,G,h,A,b)
# print sol['x']
# w = [0,0]
# for i in xrange(len(X)):
#     w += (X[i]*(sol['x'][i]*Y[i]))[0]
# print w
# b = Y[0] - w*X[0].T
# print b

def linear_to_class_func(w):
    return lambda n: w[0] + w[1] * n[0] + w[2] * n[1]

def gaussian_to_class_func(x,y,b,alphas):
    return lambda n: sum([y[i,0]*alphas[i,0]*np.exp(-1/2*np.sum(np.square(np.subtract(n,x[i])))) for i in xrange(len(x))]) + b

def error_rate(x, y, w):
    return sum(np.array([np.sign(to_class_func(w)(n)) for n in x]) != y) / float(len(x))

def linear_get_svm_ws(x, y, c):
    n = np.shape(x)[0]
    P = np.zeros((n,n))
    k = np.dot(x, x.T) # basically the "kernel matrix"
    for i in xrange(n):
        for j in xrange(n):
            P[i,j] = y[i,0]*y[j,0]*k[i,j]
    P = matrix(P,tc='d')
    q = matrix(np.ones(n)*-1,tc='d')
    G = matrix(np.concatenate((np.eye(n), -1*np.eye(n))),tc='d')
    h = matrix(np.concatenate((np.tile(c, n), np.tile(0, n))), tc='d')
    A = matrix(y, (1, n))
    b = matrix([0.0])
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x'])
    w = [0,0]
    for i in xrange(n):
        w += (x[i]*(alphas[i,0]*y[i,0]))[0]
    zero_thresh = 1e-5
    y = np.reshape(y, (-1, 1))
    x_support = np.where(np.logical_and(alphas > zero_thresh, alphas < c), x, None)
    maxa = 1
    for i in xrange(x_support.shape[0]):
        if x_support[i,0] != None:
            if alphas[i,0] < maxa:
                maxa = alphas[i,0]
                b = y[i,0] - x_support[i]*w[0].T
    return [b[0,0],w[0,0],w[0,1]]

def gaussian_get_svm_ws(x, y, c):
    n = np.shape(x)[0]
    P = np.zeros((n,n))
    for i in xrange(n):
        for j in xrange(n):
            P[i,j] = y[i,0]*y[j,0]*np.exp(-1/2*np.sum(np.square(np.subtract(x[i],x[j]))))
    P = matrix(P,tc='d')
    q = matrix(np.ones(n)*-1,tc='d')
    G = matrix(np.concatenate((np.eye(n), -1*np.eye(n))),tc='d')
    h = matrix(np.concatenate((np.tile(c, n), np.tile(0, n))), tc='d')
    A = matrix(y, (1, n))
    b = matrix([0.0])
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x'])
    zero_thresh = 1e-5
    for alpha in alphas:
        if alpha[0] < zero_thresh:
            alpha[0] = 0
    y = np.reshape(y, (-1, 1))
    x_support = np.where(alphas < c, x, None)
    maxa = 1
    s_index = 0
    for i in xrange(x_support.shape[0]):
        if x_support[i,0] != None:
            if alphas[i,0] < maxa:
                maxa = alphas[i,0]
                s_index = i
    b = y[s_index,0] - sum([y[i,0]*alphas[i,0]*np.exp(-1/2*np.sum(np.square(np.subtract(x[s_index],x[i])))) for i in xrange(len(x))])
    return b, alphas

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()
Xa = np.matrix(train[:, 0:2].copy())
Ya = np.matrix(train[:, 2:3].copy())
#
# # Carry out training, primal and/or dual
# w = linear_get_svm_ws(Xa, Ya, 1)
# print w
b,alphas = gaussian_get_svm_ws(Xa, Ya, 0.01)
print b
# print error_rate(X,Y,w)
# Define the predictSVM(x) function, which uses trained parameters
# predictSVM = linear_to_class_func(w)
predictSVM = gaussian_to_class_func(Xa,Ya,b,alphas)

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

pl.show()
