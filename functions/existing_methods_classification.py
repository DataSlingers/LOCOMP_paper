import vimpy

import pandas as pd
import numpy as np
from LOCO_classification import *

###################
####### LOCO
###################


def LOCOSplitClass(X,Y,fit_func,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    clas=np.unique(Y)
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    clas=set(y_train)


    model = fit_func(x_train,y_train)
    prob = pd.DataFrame(model.predict_proba(x_val), columns=clas)
    resids_split = getNC(y_val, prob)
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features

    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    resids_drop,resids_drop_test = {},{}
    for idd,j in enumerate(ff):
        model_out_j = fit_func(np.delete(x_train,j,1),y_train)
        prob_j = pd.DataFrame(model_out_j.predict_proba(np.delete(x_val,j,1)), columns=clas)
        resids_drop[idd]=getNC(y_val, prob_j)

        z[idd] = resids_drop[idd] - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)

    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['z']=z


    return res
def LOCOJplusClass(X,Y,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    clas=set(Y)
    ## jacknife_plus 
    resids_LOO,predictions = np.zeros(N),np.zeros(N)

    for i in range(N):
        model = fit_func(np.delete(X,i,0),np.delete(Y,i))
        prob = pd.DataFrame(model.predict_proba(np.reshape(X[i,], (1, X[i,].size))), columns=clas)
        resids_LOO[i] = getNC(Y[i:i+1], prob)

    resids_drop = np.zeros((M,N))
    # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    
    inf_z= np.zeros((len(ff),3))
    for idd,j in enumerate(ff):
        out_j = np.zeros(N)
        for i in range(N):
            X_i = np.delete(X,i,0)
            y_i = np.delete(Y,i)
            model_out_j = fit_func(np.delete(X_i,j,1),y_i) 
            prob_j = pd.DataFrame(model_out_j.predict_proba(np.delete(X[i:(i+1),],j,1)), columns=clas)
        # LOO residual 
        resids_drop= np.abs(Y - out_j)
        ## redisual difference on second 
        z[idd] = resids_drop - resids_LOO
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = ([np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)])
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res

def LOCOsplitMPClass(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    ## fit model on first part, predict on second 
    clas=set(y_train)
    #### fit MP model on 1st part 
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(x_train,y_train,x_val,n_ratio,m_ratio,B,fit_func)
    ##############
    ## average     
    predictions = pd.DataFrame(predictions.mean(0), columns=clas)
    ## get nonconformal on second 
    resids_split = getNC(y_val, predictions)
    
    
    #Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),3))
    z={}
    quantile_z=np.zeros((len(ff),2))    
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        [out_js,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(x_train,j,1),y_train,np.delete(x_val,j,1),n_ratio,m_ratio,B,fit_func)
        out_j = pd.DataFrame(out_js.mean(0), columns=clas)

        resids_drop=getNC(y_val, out_j)
        z[idd] = resids_drop - resids_split

        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = [np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)]
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res


    ## Feature CI with jackknife+
def LOCOJplusMPClass(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    clas=set(Y)
    #### fit MP model on 1st part 
   ## jacknife_plus 
    resids_LOO = np.zeros(N)

    for i in range(N):        
        [predictions,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(X,i,0),np.delete(Y,i),np.reshape(X[i,], (1, X[i,].size)),n_ratio,m_ratio,B,fit_func)
        ## average 
        predictions = pd.DataFrame(predictions.mean(0), columns=clas)
        resids_LOO[i] = getNC(Y[i:i+1], predictions)

    # Re-fit after dropping each feature
    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    z={}
    quantile_z=np.zeros((len(ff),2))
    inf_z= np.zeros((len(ff),3))
    for idd,j in enumerate(ff):
        out_j = []
        for i in range(N):
            X_i = np.delete(X,i,0)
            y_i = np.delete(Y,i)
            [out_js,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(X_i,j,1),y_i,np.delete(X[i:(i+1),],j,1),n_ratio,m_ratio,B,fit_func)
            out_j.append(out_js.mean(0)[0])
        out_j=pd.DataFrame(out_j, columns=clas)

        # LOO residual 
        resids_drop = getNC(Y, out_j)    
        z[idd] = resids_drop - resids_LOO
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = [np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)]
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res
       
    
##################################
## other existing methods
###################################





def vimeeClass(X,Y,fit_fun,alpha,measure_type='auc',selected_features=[]):
    N = len(X)
    M=len(X[0])

    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations

    folds_outer = np.random.choice(a = np.arange(2), size = N, replace = True, p = np.array([0.5, 0.5]))
    folds_outer## fit the full regression
    # cv_full.fit(x[folds_outer == 1, :], y[folds_outer == 1])
    # full_fit = cv_full.best_estimator_.predict(x[folds_outer == 1, :])
    x_1=X[idx_I,:]
    y_1=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_0 = X[out_I]
    y_0 = Y[out_I]
    ## prediction on x1 use x1
    clas=set(y_1)
    folder = np.zeros(N)
    folder[idx_I]=1
    full_fit_model = fit_fun(x_1,y_1)
    prob = full_fit_model.predict_proba(x_1)[:,1]
    if len(selected_features)==0:
        selected_features = range(M)

    res=[]

    for i in selected_features:
        x_small = np.delete(x_0, i, 1) # delete the columns in s
        small_fit_model = fit_fun(x_small,y_0)
        small_prob =  small_fit_model.predict_proba(x_small)[:,1]
        if measure_type=='accuracy':
            prob=1*(prob>=0.5)
            small_prob=1*(small_prob>=0.5)
        vimp_precompute = vimpy.vim(y = Y, x = X, s = i, f = prob, r = small_prob,
                            measure_type = measure_type, folds = folder)
        vimp_precompute.get_point_est()
            ## get the influence function estimate
        vimp_precompute.get_influence_function()
        

