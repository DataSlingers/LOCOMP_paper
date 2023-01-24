
import pandas as pd
import numpy as np
from LOCO_classification import *

def validateLOCOMPClass(X,Y,X1,Y1, n_ratio,m_ratio,B,fit_func, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    clas=np.unique(Y)
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(X,Y,np.vstack((X,X1)),n_ratio,m_ratio,B,fit_func)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]

    # Re-fit after dropping each feature
    with_j,out_j = np.zeros((N,len(clas))),np.zeros((N,len(clas)))
    y_new =[]
    zeros=False


    diff=[]
    #############################
    ## Find LOO
    ############################
    for i in range(N):
        ## find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        
        
        #####################
        ###### estimate B
        for _ in range(10):
            sel_2 = sample(b_keep,2)
            diff.append(np.square(predictions_train[sel_2[0],i] - predictions_train[sel_2[1],i]).sum())
         ##################################

        if len(b_keep)>0:
            with_j[i]= predictions_train[b_keep,i].mean(0) 
    
    with_j = pd.DataFrame(with_j, columns=clas)
   
    resids_LOO = getNC(Y, with_j)

    ################################        
    ######## FIND LOCO    
    #############################
    

    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    z={}
    resids_LOCOs={}
    inf_z = np.zeros((len(ff),4))
    for idd,j in enumerate(ff):
        out_j = np.zeros((N,len(clas)))
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            out_j[i] = predictions_train[b_keep_f,i].mean(0)
        out_j = pd.DataFrame(out_j, columns=clas)
        resids_LOCO = getNC(Y, out_j)

        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        
        resids_LOCOs[idd] = resids_LOCO.copy()
        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =bonf)
    #################################
    ## Caculate target 
    ##################################
 
    
    uhat_test = predictions_test.mean(0)
    uhat_test = pd.DataFrame(uhat_test, columns=clas)
    var,target,err_j_test,err_test,err11,err12,err21,err22={},{},{},{},{},{},{},{}
    stability_err,stability={},{}
     
    mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    uhat1_test=pd.DataFrame(predictions_test[mps1,:].mean(0), columns=clas)
    uhat2_test=pd.DataFrame(predictions_test[mps2,:].mean(0), columns=clas)
    
    for idd,j in enumerate(ff): ## ff include feature of interest

        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0)
        uhat_j_test = pd.DataFrame(uhat_j_test, columns=clas)
        
        err_j_test[idd] = getNC(Y1, uhat_j_test)
        err_test[idd] = getNC(Y1, uhat_test)
        target[idd] =np.mean(err_j_test[idd]-err_test[idd])
        var[idd]=np.std(err_j_test[idd]-err_test[idd])

    ###########################
    res= {}
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z
    res['variance']=var
    res['resids_LOO']=resids_LOO
    res['resids_LOCO']=resids_LOCOs
    res['err1'] = err_j_test
    res['err2'] = err_test
    res['diff']=diff   
    return res
 
def validateLOCOSplitMPClass(X,Y,X1, Y1, n_ratio,m_ratio,B,fit_func, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    clas=np.unique(Y)
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    clas=set(y_train)
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(x_train,y_train,np.vstack((x_val,X1)),n_ratio,m_ratio,B,fit_func)
    predictions_val = predictions[:,:len(x_val)].mean(0)
    predictions_val = pd.DataFrame(predictions_val, columns=clas)
    predictions_test = predictions[:,len(x_val):].mean(0)
    predictions_test = pd.DataFrame(predictions_test, columns=clas)

    resids_split = getNC(y_val, predictions_val)
    resids_split_test = getNC(Y1, predictions_test)
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
        
    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    resids_drop,resids_drop_test={},{}
    for idd,j in enumerate(ff):
        [out_js,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(x_train,j,1),y_train,np.delete(np.vstack((x_val,X1)),j,1),n_ratio,m_ratio,B,fit_func)

        out_j_val = out_js[:,:len(x_val)].mean(0)
        out_j_test = out_js[:,len(x_val):].mean(0)
        
        out_j_val = pd.DataFrame(out_j_val, columns=clas)
        out_j_test= pd.DataFrame(out_j_test, columns=clas)
        
        resids_drop[idd]=getNC(y_val, out_j_val)
        resids_drop_test[idd]=getNC(Y1, out_j_test)

        
        z[idd] = resids_drop[idd] - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
       

    #################################
    ## Caculate target 
    ##################################

    
    
    target,var={},{}
    uhat_test = predictions_test
    var,target,err_j_test,err_test={},{},{},{}
    for idd,j in enumerate(ff): ## ff include feature of interest
        target[idd] =np.mean(resids_drop_test[idd]-resids_split_test)
        var[idd]=np.std(resids_drop_test[idd]-resids_split_test)
        
    ###########################
    res= {}
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z
    res['variance']=var
    res['resids_LOCO']=resids_drop
    res['resids_LOO']=resids_split
    res['err1'] = resids_drop_test
    res['err2'] = resids_split_test

    
    return res   
def validateLOCOSplitClass(X,Y,X1,Y1, fit_func,  selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
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
    prob_test = pd.DataFrame(model.predict_proba(X1), columns=clas)
    resids_split = getNC(y_val, prob)
    resids_split_test = getNC(Y1, prob_test)
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
        prob_j_test = pd.DataFrame(model_out_j.predict_proba(np.delete(X1,j,1)), columns=clas)
        resids_drop[idd]=getNC(y_val, prob_j)
        resids_drop_test[idd]=getNC(Y1, prob_j_test)

        
        z[idd] = resids_drop[idd] - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        
    

    #################################
    ## Caculate target 
    ##################################

    
    
    target,var={},{}
    for idd,j in enumerate(ff): ## ff include feature of interest
        target[idd] =np.mean(resids_drop_test[idd]-resids_split_test)
        var[idd]=np.std(resids_drop_test[idd]-resids_split_test)
        
    ###########################
    res= {}
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z
    res['variance']=var
    res['resids_LOCO']=resids_drop
    res['resids_LOO']=resids_split
    res['err1'] = resids_drop_test
    res['err2'] = resids_split_test

    
    return res