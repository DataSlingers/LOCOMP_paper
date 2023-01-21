import numpy as np
import pandas as pd
from sklearn import preprocessing

    
def SimuAutoregressive(N,M,N1,SNR, seed=1):
    M1=int(0.05*M)
    np.random.seed(seed)
    mu = [0]*M
    cov = [[0]*M for _ in range(M)]
    for i in range(M-1):
        cov[i][i+1]=0.5
        cov[i+1][i]=0.5
    cov=np.array(cov)
    np.fill_diagonal(cov, 1)

    X = np.random.multivariate_normal(mu, cov,N)
    np.random.rand(seed*2+1)
    X1 =  np.random.multivariate_normal(mu, cov,N1)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))

    Y = np.dot(X,beta)++np.random.normal(0, 1,N)

    Y1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)
    return [X,Y,X1,Y1]

def SimuLinear(N,M,N1,SNR, seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    M1=int(0.05*M)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))

    Y = np.dot(X,beta)+np.random.normal(0, 1,N)
    Y1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)

    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)

    return [X,Y,X1,Y1]

 
def SimuNonlinear(N,M,N1,SNR, seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.seed(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    beta = np.append([SNR],np.random.normal(5,1,5))

    Y = beta[0]*X[:,0]*(X[:,0]<2 &(X[:,0]>-2))+beta[1]*X[:,1]*(X[:,1]<0)+(X[:,2]>0)*X[:,2]*beta[2]+(X[:,3]>0)*X[:,3]*beta[3]+ (np.sign(X[:,4]))*beta[4]+np.random.normal(0, 1,N)
    Y1 = beta[0]*X1[:,0]*(X1[:,0]<2 &(X1[:,0]>-2))+beta[1]*X1[:,1]*(X1[:,1]<0)+(X1[:,2]>0)*X1[:,2]*beta[2]+(X1[:,3]>0)*X1[:,3]*beta[3]+ (np.sign(X1[:,4]))*beta[4]+np.random.normal(0, 1,N1)

    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)

    return [X,Y,X1,Y1]
