import numpy as np
import pandas as pd
from scipy.stats import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy.random as r
from scipy.stats import *
from random import sample

from joblib import Parallel, delayed

def LOCOMPClass(X,Y, n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    clas=np.unique(Y)
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(X,Y,X,n_ratio,m_ratio,B,fit_func)
    predictions_train = predictions

    zeros=False


    diff=[]
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))

    #############################
    ## Find LOO
    ############################
    for i in range(N):
        #####################
        ###### estimate B
        sel_2 = np.array(sample(list(b_keep[i]),20))
        sel_2.shape = (2,10)
        diff.append(np.square(predictions_train[sel_2[0],i][:,0] - predictions_train[sel_2[1],i][:,0]).mean())

    with_j = map(lambda i: predictions_train[b_keep[i],i].mean(0),range(N))
    with_j = pd.DataFrame(list(with_j), columns=clas)
    resids_LOO = getNC(Y, with_j)

    ################################
    ######## FIND LOCO
    #############################
    def get_loco(i,j):
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        return predictions_train[b_keep_f,i].mean(0)

    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    results = Parallel(n_jobs=-1)(delayed(get_loco)(i,j) for i in range(N) for j in range(M))
    ress = pd.DataFrame(results)
    ress['i'] = np.repeat(range(N),M)
    ress['j'] = np.tile(range(M),N)
    ress['true_y'] = np.repeat(Y,M)
    ress['resid_loco'] = getNC(ress['true_y'], ress[[0,1]])
    ress['resid_loo'] = np.repeat(resids_LOO,M)
    ress['zz'] = ress['resid_loco'] -ress['resid_loo']


    inf_z = np.zeros((len(ff),4))
    for idd,j in enumerate(ff): 
        inf_z[idd] = ztest(ress[ress.j==idd].zz,alpha,MM=len(ff),bonf_correct =bonf)
    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['info']=ress
    res['diff']=diff
    return res
def getNC(true_y,prob,method = 'prob1'):
    if method=='prob2':
        if len(true_y)==1:
            true_y=true_y[0]
            py=prob[true_y]
            pz = max(prob.drop(true_y,axis=1))
            nc = (1- py+pz)/2
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            pz=[max(prob.iloc[i].drop(true_y[i])) for i in range(len(true_y))] ## max prob of other label 
            nc = [(1- py[i]+pz[i])/2 for i in range(len(py))]
    if method=='prob1':
        if len(true_y)==1:
            true_y=int(true_y[0])
            py=prob[true_y]
            nc = (1- py)
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            nc = [(1- py[i]) for i in range(len(py))]
    return np.array(nc)


def buildMPClass(X,Y,n_ratio,m_ratio):
    N = len(X)
    M = len(X[0])
    n = np.int(np.ceil(n_ratio * N))
    m = np.int(np.ceil(m_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    #3 stratified sampling
    Y_pd=pd.DataFrame(Y.reshape((len(Y),1)))

    idx_I =Y_pd.groupby(0, group_keys=False).apply(lambda x: x.sample(frac=n_ratio))
    idx_I = np.sort(list(idx_I.index)) # stratified sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]



def predictMPClass(X,Y,X1, n_ratio,m_ratio,B,fit_func):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    clas=set(Y)

    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):
        [idx_I,idx_F,x_mp,y_mp] = buildMPClass(X,Y,n_ratio,m_ratio)
        model = fit_func(x_mp,y_mp)
        prob = pd.DataFrame(model.predict_proba(X1[:, idx_F]), columns=set(y_mp))
        for i in (clas):
            if i not in prob.columns:
                prob[i]=0
    ############################################
        predictions.append(np.array(prob))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]



## mean inference 
      
def ztest(z,alpha,MM=1,bonf_correct=True):
    l = len(z)
    s = np.std(z)
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/s*np.sqrt(l))
    pval2 = 2*(1-norm.cdf(np.abs(m/s*np.sqrt(l))))
    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
    q = norm.ppf(1-alpha/2)
    left  = m - q*s/np.sqrt(l)
    right = m + q*s/np.sqrt(l)
    return [pval1,pval2, left,right]



