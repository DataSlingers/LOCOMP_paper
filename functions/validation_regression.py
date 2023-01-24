
import pandas as pd
import numpy as np
from functions.LOCO_regression import *

def validateLOCOMPReg(X,Y,X1, Y1, n_ratio,m_ratio,B,fit_func,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,np.vstack((X,X1)),n_ratio,m_ratio,B,fit_func)
    predictions_train = predictions[:,:N]
    predictions_test = predictions[:,N:]
    
    
    
    
    # Re-fit after dropping each feature
    resids_LOO,resids_LOCO = np.zeros(N),np.zeros(N)
    resids_LOO_sq,resids_LOCO_sq = np.zeros(N),np.zeros(N)
    y_new =np.zeros((N,N1), dtype=float)
    zeros=False

    #############################
    ## Find LOO
    ############################
    diff= []
    for i in range(N):
        ## find MP has no i but has j
        b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        
        #####################
        ###### estimate B
        for _ in range(10):
            sel_2 = sample(b_keep,2)
            diff.append(np.square(predictions_train[sel_2[0],i] - predictions_train[sel_2[1],i]))
         ##################################
                  
        if len(b_keep)>0:
            resids_LOO[i]= np.abs(Y[i] - predictions_train[b_keep,i].mean())
            resids_LOO_sq[i]= np.square(Y[i] - predictions_train[b_keep,i].mean())
    ################################
    
    
        ################################
    ######## FIND LOCO
    #############################
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z,inf_z_sq = np.zeros((len(ff),4)), np.zeros((len(ff),4))
    z,z_sq={},{}
    resids_LOCOs,resids_LOCOs_sq={},{}

    for idd,j in enumerate(ff):
        print(idd,j)
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= np.abs(Y[i] - predictions_train[b_keep_f,i].mean())
            resids_LOCO_sq[i]= np.square(Y[i] - predictions_train[b_keep_f,i].mean())
        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()


        zz = resids_LOCO_sq - resids_LOO_sq
        z_sq[idd] = zz[~np.isnan(zz)]
        resids_LOCOs_sq[idd] = resids_LOCO_sq.copy()
        
        
        
        
        
        
        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
            inf_z_sq[idd] = ztest(z_sq[idd],alpha,MM=len(ff),bonf_correct =True)

    #################################
    ## Caculate target
    ##################################

    ## average test yhat
    #########################################
    uhat_test=predictions_test.mean(0)
    mps1 = list(set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
    mps2 = list(set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
    uhat1_test=predictions_test[mps1,:].mean(0)
    uhat2_test=predictions_test[mps2,:].mean(0)
    ## leave-j-out yhat
    var,target,err1,err2,err11,err12,err21,err22={},{},{},{},{},{},{},{}
    var_sq,target_sq,err1_sq,err2_sq,err11_sq,err12_sq,err21_sq,err22_sq={},{},{},{},{},{},{},{}
    stability_err_sq,stability_err,stability,stability_sq={},{},{},{}
    for idd,j in enumerate(ff): ## ff include feature of interest
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))
        uhat_j_test= predictions_test[b_keep_f,:].mean(0) ## TEST ERROR WITHOUT J
        
        b_keep_f1 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,1])).reshape(-1)))
        uhat1_j_test=predictions_test[b_keep_f1,:].mean(0)
        b_keep_f2 = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,2])).reshape(-1)))
        uhat2_j_test=predictions_test[b_keep_f2,:].mean(0)
        
        
        err1[idd] = np.abs(Y1-uhat_j_test)
        err2[idd] = np.abs(Y1-uhat_test)
        target[idd] = np.mean(err1[idd]-err2[idd])
        var[idd]= np.std(err1[idd]-err2[idd])
        
        err11[idd] = np.abs(Y1-uhat1_j_test)
        err12[idd] = np.abs(Y1-uhat1_test)
        err21[idd] = np.abs(Y1-uhat2_j_test)
        err22[idd] = np.abs(Y1-uhat2_test)
        stability_err[idd] = err11[idd] - err12[idd] - err21[idd] + err22[idd]
        stability[idd]= np.std(stability_err[idd])
        

        err1_sq[idd] = (Y1-uhat_j_test)**2
        err2_sq[idd] = (Y1-uhat_test)**2
        target_sq[idd] = np.mean(err1_sq[idd]-err2_sq[idd])
        var_sq = np.std(err1_sq[idd]-err2_sq[idd])
        
        err11_sq[idd] = (Y1-uhat1_j_test)**2
        err12_sq[idd] = (Y1-uhat1_test)**2
        err21_sq[idd] = (Y1-uhat2_j_test)**2
        err22_sq[idd] = (Y1-uhat2_test)**2
        stability_err_sq[idd] = err11_sq[idd] - err12_sq[idd] - err21_sq[idd] + err22_sq[idd]
        stability_sq[idd]= np.std(stability_err_sq[idd])


    ###########################
    res= {}
    res['diff']=diff
    res['stability']=stability
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z
    res['variance']=var
    res['resids_LOO']=resids_LOO
    res['resids_LOCO']=resids_LOCOs
    res['err1'] = err1
    res['err2'] = err2
    res['stability_err'] = stability_err
    

    res['stability_sq']=stability_sq
    res['stability_err_sq'] = stability_err_sq
    res['target_sq']=target_sq
    res['loco_ci_sq']=inf_z_sq
    res['z_sq']=z_sq
    res['variance_sq']=var_sq
    res['resids_LOO_sq']=resids_LOO_sq
    res['resids_LOCO_sq']=resids_LOCOs_sq
    
    return res



def validateLOCOSplitReg(X,Y,X1, Y1,fit_func,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)

    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    
    
    ## fit model on first part, predict on second 
    prediction = fit_func(x_train,y_train,np.vstack((x_val,X1)))
    predictions  = prediction[:len(y_val)]
    predictions_test = prediction[len(y_val):]
    ## get residual on second 
    resids_split = np.abs(y_val - predictions)
    resids_split_sq = (y_val - predictions)**2

    resids_split_test = np.abs(Y1 - predictions_test)
    resids_split_test_sq = np.square(Y1 - predictions_test)

    # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = range(M)
    else:
        ff=selected_features
    inf_z,inf_z_sq= np.zeros((len(ff),4)),np.zeros((len(ff),4))
    
    uhat_j_test={}
    
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        out_js = fit_func(np.delete(x_train,j,1),y_train,np.delete(np.vstack((x_val,X1)),j,1)) 
        
        out_j  = out_js[:len(y_val)]
        uhat_j_test[idd] = out_js[len(y_val):]
    
        
        resids_drop=np.abs(y_val - out_j)
        resids_drop_sq = np.mean((y_val - out_j)**2)
        z = resids_drop - resids_split
        z_sq= resids_drop_sq - resids_split_sq
        inf_z[idd] = ztest(z,alpha,bonf_correct =True)
        inf_z_sq[idd] = ztest(z_sq,alpha,bonf_correct =True)

        
    uhat_test=predictions_test ## TEST ERROR WITH J 
    ## leave-j-out yhat
    target,target_raw,err1,err2,err1_raw,err2_raw={},{},{},{},{},{}
    target_sq,target_raw_sq,err1_sq,err2_sq,err1_raw_sq,err2_raw_sq={},{},{},{},{},{}
    for j in ff: ## ff include feature of interest

        
        err1[idd] = np.abs(Y1-uhat_j_test[idd])
        err2[idd] = np.abs(Y1-uhat_test)
        target[idd] = np.mean(err1[idd]-err2[idd])

        err1_sq[idd] = (Y1-uhat_j_test[idd])**2
        err2_sq[idd] = (Y1-uhat_test)**2
        target_sq[idd] = np.mean(err1_sq[idd]-err2_sq[idd])


    ###########################
    res= {}
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z

    res['err1'] = err1
    res['err2'] = err2
    res['resids_LOCO']=resids_drop
    res['resids_LOO']=resids_split

    res['err1_sq'] = err1_sq
    res['err2_sq'] = err2_sq
    res['resids_LOCO_sq']=resids_drop
    res['resids_LOO_sq']=resids_split
    res['target_sq']=target_sq
    res['loco_ci_sq']=inf_z_sq
    res['z_sq']=z_sq


        
    return res 
    

def validateLOCOSplitMPReg(X,Y,X1, Y1, n_ratio,m_ratio,B,fit_func,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    [predictions,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,np.vstack((x_val,X1)),n_ratio,m_ratio,B,fit_func)
    predictions_val = predictions[:,:len(x_val)].mean(0)
    predictions_test = predictions[:,len(x_val):].mean(0)

    # Re-fit after dropping each feature
    resids_split = np.abs(y_val - predictions_val)    
    resids_split_sq =np.square(y_val - predictions_val)
    resids_split_test = np.abs(Y1 - predictions_test)    
    resids_split_test_sq = np.square(Y1 - predictions_test)    

    #Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features

    inf_z,inf_z_sq = np.zeros((len(ff),4)), np.zeros((len(ff),4))
    z,z_sq={},{}
    resids_drop,resids_drop_sq={},{}
    resids_drop_test_sq,resids_drop_test={},{}
    out_j_test={}
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        [out_js,in_mp_obs,in_mp_feature]= predictMP(np.delete(x_train,j,1),y_train,np.delete(np.vstack((x_val,X1)),j,1),n_ratio,m_ratio,B,fit_func)
        out_j_val = out_js[:,:len(x_val)].mean(0)
        out_j_test[idd] = out_js[:,len(x_val):].mean(0)
        
        resids_drop_test[idd]=np.abs(Y1 - out_j_test[idd])
        resids_drop_test_sq[idd]=np.square(Y1 - out_j_test[idd])
        
        resids_drop[idd]=np.abs(y_val - out_j_val)
        resids_drop_sq[idd]= np.square(y_val - out_j_val)
        z[idd] = resids_drop[idd] - resids_split
        z_sq[idd]= resids_drop_sq[idd] - resids_split
        
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        inf_z_sq[idd] = ztest(z_sq[idd],alpha,MM=len(ff),bonf_correct =True)
        
    #################################
    ## Caculate target
    ##################################

    ######################################
    ## average test yhat
    #########################################
    
    var,target={},{}
    var_sq,target_sq={},{}
    
    for idd,j in enumerate(ff): ## ff include feature of interest
         
        target[idd] = np.mean(resids_drop_test[idd]-resids_split_test)
        var[idd]= np.std(resids_drop_test[idd]-resids_split_test)

        target_sq[idd] = np.mean(resids_drop_test_sq[idd]-resids_split_test_sq)
        var_sq = np.std(resids_drop_test_sq[idd]-resids_split_test_sq)


    ###########################
    res= {}
    res['target']=target
    res['loco_ci']=inf_z
    res['z']=z
    res['variance']=var
    res['resids_split']=resids_split
    res['resids_drop']=resids_drop
    res['err1'] = resids_drop_test
    res['err2'] = resids_split_test



    res['err1_sq'] = resids_drop_test_sq
    res['err2_sq'] = resids_split_test_sq
    res['target_sq']=target_sq
    res['loco_ci_sq']=inf_z_sq
    res['z_sq']=z_sq
    res['variance_sq']=var_sq
    res['resids_split_sq']=resids_split_sq
    res['resids_drop_sq']=resids_drop_sq
    return res 
