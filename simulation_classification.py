import numpy as np
import pandas as pd
from sklearn import preprocessing

def SimuLinearClass(N,M,N1,SNR,M1=5,seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    M1=int(0.05*M)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))
    pr = np.dot(X,beta)+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)

    pr1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)
    return (X,Y,X1,Y1)

def SimuNonlinearClass(N,M,N1,SNR,M1=5,seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    beta = np.append([SNR],np.random.normal(5,1,5))
    pr = beta[0]*X[:,0]*(X[:,0]<2 &(X[:,0]>-2))+beta[1]*X[:,1]*(X[:,1]<0)+(X[:,2]>0)*X[:,2]*beta[2]+(X[:,3]>0)*X[:,3]*beta[3]+ (np.sign(X[:,4]))*beta[4]+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)
    pr1 = beta[0]*X1[:,0]*(X1[:,0]<2 &(X1[:,0]>-2))+beta[1]*X1[:,1]*(X1[:,1]<0)+(X1[:,2]>0)*X1[:,2]*beta[2]+(X1[:,3]>0)*X1[:,3]*beta[3]+ (np.sign(X1[:,4]))*beta[4]+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)
    return (X,Y,X1,Y1)

def SimuCorrelatedClass(N,M,N1,SNR, seed=1):
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
    np.random.seed(seed*2+1)
    X1 =  np.random.multivariate_normal(mu, cov,N1)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))
    pr = np.dot(X,beta)+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)
    pr1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)

    return [X,Y,X1,Y1]
