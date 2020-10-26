import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
d,n= x.shape
i = 0 #Index of the image to be visualized
plt.imshow(np.reshape(x[:,i], (int(np.sqrt(d)),int(np.sqrt(d)))))

ntrain = 2000
ntest = n - ntrain
xtrain = x[:,0:ntrain]
ytrain = y[:,0:ntrain]
xtest = x[:,ntrain:]
ytest = y[:,ntrain:]
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
xtrain=np.row_stack((np.ones(2000),xtrain))
xtest=np.row_stack((np.ones(1000),xtest))


A = (xtrain*ytrain).T
print(A.shape)
reg = 10
# expxy = lambda x: np.exp(-yt*np)

exp_theta = lambda theta: np.exp(-ytrain*np.dot(np.reshape(theta,()),xtrain))
cost1 = lambda theta: np.sum(np.log(1+exp_theta(theta)) ) + reg * ( np.linalg.norm(theta)**2 )
grad1 = lambda theta: np.sum(-exp_theta(theta) / (1+exp_theta(theta))*ytrain*xtrain,axis=1 ) + 2*reg*theta
hess1 = lambda theta: np.dot( (exp_theta(theta)/(1+exp_theta(theta))**2) * xtrain, xtrain.T ) + 2*reg*np.identity(theta.shape[0])



pot = lambda z: np.log(1 + np.exp(-z))
dpot = lambda z: -1 / (np.exp(z) + 1)
ddpot = lambda z: np.exp(z) / (np.exp(z)+1)**2
cost = lambda theta: sum( pot( np.dot(A,theta) ) ) + reg * ( np.linalg.norm(theta)**2 )
grad = lambda theta: np.dot(A.T, dpot( np.dot(A,theta)) ) + 2*reg*theta
hess = lambda theta: np.dot(np.dot(A.T, np.diag( ddpot( np.dot(A,theta)) ) ), A) + 2*reg*np.identity(theta.shape[0])
L = 0.25*(np.linalg.norm(A)**2) + 2*reg


def NewtonRaphson(x0):
    xk = x0
    niter = 0
    while True:
        xk1 = xk - np.dot(np.linalg.inv(hess(xk)),grad(xk))
        error = abs( cost(xk1) - cost(xk) ) / cost(xk)
        xk = xk1
        niter = niter+1
        print('cost',cost(xk))
        if(error < 1e-6):
            break
    return xk, niter


# iteration
    x0 = np.zeros(785)
    theta,niter=NewtonRaphson(x0)
    print('niter is',niter)

    plt.imshow(np.reshape(theta[1:], (int(np.sqrt(d)),int(np.sqrt(d)))))

    # calculate the error rate
rate = (np.sign(np.dot(theta,xtest)) == ytest).sum(axis=1)/ntest
print('error rate is ', 1-rate)


# find 20 max cost function
a = ( np.dot(theta,xtest)*ytest ).reshape(-1)
top20=np.argsort(a)[0:20]
top20.shape


fig, axs = plt.subplots(4, 5)
num = -1
for i in range(0,4):
    for j in range(0,5):
        num = num+1
        axs[i, j].imshow(np.reshape(xtest[1:,top20[num]], (int(np.sqrt(d)),int(np.sqrt(d)))))

        
