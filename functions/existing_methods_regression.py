import vimpy

import pandas as pd
import numpy as np
from LOCO_regression import *

###################
####### LOCO
###################

def LOCOSplitReg(X,Y,fit_func,selected_features=[0],alpha=0.1,bonf=True):
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
    predictions = fit_func(x_train,y_train,(x_val))
    ## get residual on second 
    resids_split = np.abs(y_val - predictions)
 # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        out_j = fit_func(np.delete(x_train,j,1),y_train,np.delete(x_val,j,1))
        resids_drop=np.abs(y_val - out_j)
        z[idd] = resids_drop - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = ([np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)])
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res


def LOCOJplusReg(X,Y,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    ## jacknife_plus 
    resids_LOO,predictions = np.zeros(N),np.zeros(N)

    for i in range(N):
        predicts = fit_func(np.delete(X,i,0),np.delete(Y,i),np.reshape(X[i,], (1, X[i,].size)))
        predictions[i]=predicts[0]
        resids_LOO[i] = np.abs(Y[i] - predicts[0])

    res_drop = np.zeros((M,N))
    # Re-fit after dropping each feature
    pval=[]
    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    z={}
    quantile_z=np.zeros((len(ff),2))
    inf_z= np.zeros((len(ff),3))
    for idd,j in enumerate(ff):
        out_j = np.zeros(N)
        for i in range(N):
            X_i = np.delete(X,i,0)
            y_i = np.delete(Y,i)
            out_j[i] = fit_func(np.delete(X_i,j,1),y_i,np.delete(X[i:(i+1),],j,1))[0]
        # LOO residual 
        resids_drop= np.abs(Y - out_j)
        ## redisual difference on second 
        z[idd] = resids_drop -resids_LOO 
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = ([np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)])
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res




############################
## LOCO on MP ensemble
#############################
def LOCOsplitMPReg(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False):
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
    #### fit MP model on 1st part 
    [predictions,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,x_val,n_ratio,m_ratio,B,fit_func)
    ## average 
    predictions = predictions.mean(0)    
    ## get residual on second 
    resids_split = np.abs(y_val - predictions)    
    
    #Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        [out_js,in_mp_obs,in_mp_feature]= predictMP(np.delete(x_train,j,1),y_train,np.delete(x_val,j,1),n_ratio,m_ratio,B,fit_func)
        out_j = out_js.mean(0)
        resids_drop=np.abs(y_val - out_j)
        z[idd] = resids_drop - resids_split

        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = [np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)]
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res

    ## Feature CI with jackknife+
def LOCOJplusMPReg(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    ## jacknife_plus 
    resids_LOO,predictions = np.zeros(N),np.zeros(N)

    for i in range(N):        
        [predicts,in_mp_obs,in_mp_feature]= predictMP(np.delete(X,i,0),np.delete(Y,i),np.reshape(X[i,], (1, X[i,].size)),n_ratio,m_ratio,B,fit_func)
        ## average 
        predicts = predicts.mean(0)[0]         
        predictions[i]=predicts  
        resids_LOO[i] = np.abs(Y[i] - predicts)

    # Re-fit after dropping each feature
    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    z={}
    quantile_z=np.zeros((len(ff),2))
    inf_z= np.zeros((len(ff),3))
    for idd,j in enumerate(ff):
        out_j = np.zeros(N)
        for i in range(N):
            X_i = np.delete(X,i,0)
            y_i = np.delete(Y,i)
            
            [out_js,in_mp_obs,in_mp_feature]= predictMP(np.delete(X_i,j,1),y_i,np.delete(X[i:(i+1),],j,1),n_ratio,m_ratio,B,fit_func)
            out_j[i] = out_js.mean(0)[0]
        # LOO residual 
        resids_drop = np.abs(Y - out_j)        
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


def vimee(X,Y,fit_fun,alpha,measure_type='r_squared',selected_features=[]):
    np.random.seed()
    N = len(X)
    M=len(X[0])
    folds_outer = np.random.choice(a = np.arange(2), size = N, replace = True, p = np.array([0.5, 0.5]))
    ## fit the full regression
    # cv_full.fit(x[folds_outer == 1, :], y[folds_outer == 1])
    # full_fit = cv_full.best_estimator_.predict(x[folds_outer == 1, :])
    x_1=X[folds_outer==1,:]
    y_1=Y[np.ix_(folds_outer==1)]
    x_0 = X[folds_outer==0,:]
    y_0 = Y[folds_outer==0]
    ## prediction on x1 use x1
    full_fit = np.array(fit_fun(x_1,y_1,x_1))
    res=[]

    if len(selected_features)==0:
        selected_features = range(M)



    for i in selected_features:
        x_small = np.delete(x_0, i, 1) # delete the columns in s
        small_fit = np.array(fit_fun(x_small,y_0,x_small))
        vimp_precompute = vimpy.vim(y = Y, x = X, s = i, f = full_fit, r = small_fit,
                            measure_type = measure_type, folds = folds_outer)
        vimp_precompute.get_point_est()
            ## get the influence function estimate
        vimp_precompute.get_influence_function()
        ## get a standard error
        vimp_precompute.get_se()
        ## get a confidence interval
        vimp_precompute.get_ci(level=0.9)
        vimp_precompute.hypothesis_test(alpha = 0.1, delta = 0)

        res.append([vimp_precompute.p_value_]+list(vimp_precompute.ci_[0]))
    return res


def gcm(X,Y,fit_fun,alpha,selected_features=[]):
    N = len(X)
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    M=len(X[0])
    if len(selected_features)==0:
        selected_features = range(M)

    ## GCM
    res = []
    for i in selected_features:
        x_small = np.delete(x_train, i, 1)
        f_fit = np.array(fit_fun(x_small,y_train,np.delete(x_val, i, 1)))
        eps_f = y_val - f_fit

        y_train_g =x_train[:,i]
        y_val_g=x_val[:,i]

        g_fit = np.array(fit_fun(np.delete(x_train, i, 1),
                        y_train_g,
                        np.delete(x_val, i, 1)))
        eps_g = y_val_g - g_fit
        r = eps_f * eps_g

        m = (np.mean(r))
        sd = np.sqrt(np.mean(r**2) - (np.mean(r))**2)/np.sqrt(len(x_val))
        gcm = abs(m/sd)
        pval = 2*(1-norm.cdf(gcm))
        q = norm.ppf(1-alpha/2)
        left  = abs(m) - q*sd
        right = abs(m) + q*sd
        res.append([pval,left,right])
    return res
