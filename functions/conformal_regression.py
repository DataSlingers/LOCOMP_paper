###############################
## regular conformal
################################
def splitConformalReg(X,Y,X1,fit_func,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    predictions = fit_func(x_train,y_train,(x_val))
    resids_split = np.abs(y_val - predictions)
    full_pred = fit_func(x_train,y_train,X1)
    ind_q = (np.ceil((1-alpha)*(N/2+1))).astype(int)
    return pd.DataFrame(\
                np.c_[full_pred -  np.sort(resids_split)[ind_q-1], \
                    full_pred + np.sort(resids_split)[ind_q-1]],\
                        columns = ['lower','upper'])

## cross
def crossConformalReg(X,Y,X1,fit_func,K=5,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    resids_cv=[]
    full_pred=[]
    kf = KFold(K, shuffle=True)
    for train_index, test_index in kf.split(X):
        x_train, x_val = X[train_index], X[test_index]
        y_train,y_val = Y[train_index], Y[test_index]
        predictions = fit_func(x_train,y_train,(x_val))
        resids_cv.append(np.abs(y_val - predictions))
        if len(full_pred)==0:
            
            full_pred=np.array(fit_func(x_train,y_train,X1))
        else:
            full_pred += np.array( fit_func(x_train,y_train,X1))
            print(full_pred)
    print(full_pred)
 
    full_pred=full_pred/K    
    resids_cv=np.concatenate( resids_cv, axis=0 )
    ind_q = (np.ceil((1-alpha)*(N+1))).astype(int)
    return pd.DataFrame(\
                        np.c_[full_pred -  np.sort(resids_cv)[ind_q-1], \
                        full_pred + np.sort(resids_cv)[ind_q-1]],\
                            columns = ['lower','upper'])

# J+
def jacknifePlusReg(X,Y,X1,fit_func,fun_para,alpha=0.1):
    '''
    Using mean aggregation
    '''
    N=len(X)
    N1 = len(X1)
    resids_LOO = np.zeros(N)
    new_y = np.zeros((N,N1))
    for i in range(N):
        predictions = fit_func(np.delete(X,i,0),np.delete(Y,i),np.vstack((X[i,],X1)))
        resids_LOO[i] = np.abs(Y[i] - predictions[0])
        new_y[i] = predictions[1:]

    ind_q = (np.ceil((1-alpha)*(N+1))).astype(int)
        ###############################
        # construct prediction intervals
        ###############################

    return pd.DataFrame(\
                np.c_[np.sort(new_y.T - resids_LOO,axis=1).T[-ind_q], \
                    np.sort(new_y.T + resids_LOO,axis=1).T[ind_q-1]],\
                        columns = ['lower','upper'])


## BOOTSTRAP J+
def generateBootstrapSamples(n, m, B,replace):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx = np.zeros((B, m),dtype=int)
    for b in range(B):
        sample_idx = np.random.choice(n, m,replace=replace)

        samples_idx[b, :] = sample_idx       
    return(samples_idx)



def fitBootstrapModels(X_train, Y_train, X_predict, fit_func, N, n, B,replace):
    '''
      Train B bootstrap estimators and calculate predictions on X_predict
      Return: list of matrices [M,P]
        samples_idx = B-by-n matrix, row b = indices of b-th bootstrap sample
        predictions = B-by-N1 matrix, row b = predictions from b-th bootstrap sample
          (n1=len(X_predict))
    '''
    samples_idx = generateBootstrapSamples(N, n, B,replace)
    N1 = len(X_predict)
    # P holds the predictions from individual bootstrap estimators
    predictions = np.zeros((B, N1), dtype=float)
    for b in range(B):
        predictions[b] = fit_func(X_train[samples_idx[b], :],\
                                     Y_train[samples_idx[b], ], X_predict)
    return([samples_idx, predictions])


def JplusabReg(X,Y,X1,n_ratio,B,fit_func,replace=False,alpha=0.1):
    '''
    Using mean aggregation
    '''
    N=len(X)
    N1 = len(X1)
    n = np.int(np.round(n_ratio * N))
    [boot_samples_idx,boot_predictions] = \
        fitBootstrapModels(X, Y, np.vstack((X,X1)), fit_func, N, n, B,replace)
    in_boot_sample = np.zeros((B,N),dtype=bool)
    for b in range(len(in_boot_sample)):
        in_boot_sample[b,boot_samples_idx[b]] = True
    resids_LOO = np.zeros(N)
    muh_LOO_vals_testpoint = np.zeros((N,N1))
    for i in range(N):
        b_keep = np.argwhere(~(in_boot_sample[:,i])).reshape(-1)
        if(len(b_keep)>0):
            
            resids_LOO[i] = np.abs(Y[i] - boot_predictions[b_keep,i].mean())
            muh_LOO_vals_testpoint[i] = boot_predictions[b_keep,N:].mean(0)
            
        else: # if aggregating an empty set of models, predict zero everywhere
            resids_LOO[i] = np.abs(Y[i])
            muh_LOO_vals_testpoint[i] = np.zeros(N1)
    ind_q = (np.ceil((1-alpha)*(N+1))).astype(int)
    ###############################
    # construct prediction intervals
    ###############################
        
    return pd.DataFrame(\
            np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T[-ind_q], \
                np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T[ind_q-1]],\
                    columns = ['lower','upper'])




#######################
## CONFORMAL on MP ensemble
#########################
def splitConformalMPReg(X,Y,X1,n_ratio,m_ratio,B,fit_func,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    #########
    [predictions,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,x_val,n_ratio,m_ratio,B,fit_func)
    predictions = predictions.mean(0)    

    
    #########    
    resids_split = np.abs(y_val - predictions)
    
    [full_pred,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,X1,n_ratio,m_ratio,B,fit_func)
    full_pred=full_pred.mean(0)
    ind_q = (np.ceil((1-alpha)*(N/2+1))).astype(int)
    return pd.DataFrame(\
                np.c_[full_pred -  np.sort(resids_split)[ind_q-1], \
                    full_pred + np.sort(resids_split)[ind_q-1]],\
                        columns = ['lower','upper'])


## CROSS+MP
def crossConformalMPReg(X,Y,X1,n_ratio,m_ratio,B,fit_func,K=5,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    resids_cv=[]
    full_pred=[]
    kf = KFold(K, shuffle=True)
    for train_index, test_index in kf.split(X):
        x_train, x_val = X[train_index], X[test_index]
        y_train,y_val = Y[train_index], Y[test_index]
        
        [predictions,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,x_val,n_ratio,m_ratio,B,fit_func)
        predictions = predictions.mean(0)            
        resids_cv.append(np.abs(y_val - predictions))
        if len(full_pred)==0:
            [full_pred,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,X1,n_ratio,m_ratio,B,fit_func)
            full_pred=full_pred.mean(0)
        else:
            [full,in_mp_obs,in_mp_feature]= predictMP(x_train,y_train,X1,n_ratio,m_ratio,B,fit_func)
            full_pred += full.mean(0)
    full_pred=full_pred/K    
    resids_cv=np.concatenate( resids_cv, axis=0 )
    ind_q = (np.ceil((1-alpha)*(N+1))).astype(int)
    return pd.DataFrame(\
                        np.c_[full_pred -  np.sort(resids_cv)[ind_q-1], \
                        full_pred + np.sort(resids_cv)[ind_q-1]],\
                            columns = ['lower','upper'])

## J+ MP 
def jacknifePlusbyMPReg(X,Y,X1,n_ratio,m_ratio,B,fit_func,alpha=0.1):
    '''
    Using mean aggregation
    '''
    
    N=len(X)
    N1 = len(X1)
    resids_LOO = np.zeros(N)
    new_y = np.zeros((N,N1))
    for i in range(N):
        [predictions,in_mp_obs,in_mp_feature]= predictMP(np.delete(X,i,0),np.delete(Y,i),np.vstack((X[i,],X1)),n_ratio,m_ratio,B,fit_func)
        predictions=predictions.mean(0)

        resids_LOO[i] = np.abs(Y[i] - predictions[0])
        new_y[i] = predictions[1:]

    ind_q = (np.ceil((1-alpha)*(N+1))).astype(int)
        ###############################
        # construct prediction intervals
        ###############################

    return pd.DataFrame(\
                np.c_[np.sort(new_y.T - resids_LOO,axis=1).T[-ind_q], \
                    np.sort(new_y.T + resids_LOO,axis=1).T[ind_q-1]],\
                        columns = ['lower','upper'])
###############################