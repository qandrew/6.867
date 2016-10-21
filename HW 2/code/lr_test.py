from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code
from sklearn.linear_model import LogisticRegression

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

print len(X),X[0]
print "X, Y"
print len(Y), Y[0]

# Carry out training.
lr = LogisticRegression(penalty='l2', tol=0.0001, C=1000000.0, fit_intercept=True, intercept_scaling=1, solver='liblinear', max_iter=2000)

# Define the predictLR(x) function, which uses trained parameters
### TODO ###
lr.fit(X,Y)
def predictLR(x):
    return lr.predict_proba(x)[0,1]

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show()
