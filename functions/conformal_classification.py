## J+ MP 
def jacknifePlusbyMPClass(X,Y,X1,n_ratio,m_ratio,B,fit_func,alpha=0.1):
    '''
    Using mean aggregation
    '''
    
    N=len(X)
    N1 = len(X1)
    clas=np.unique(Y)
    resids_LOO =[]
    nc_new =[]
    for i in range(N):    
        [predictions,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(X,i,0),np.delete(Y,i),np.vstack((X[i,],X1)),n_ratio,m_ratio,B,fit_func)
        predictions=predictions.mean(0)
        resids_LOO.append(getNC([Y[i]], predictions[0]))
        
        prob_new = pd.DataFrame(predictions[1:,:], columns=clas)
        resids_new = {}
        for ll in range(len(clas)):
            yi=[clas[ll]]*len(X1)
            resids_new[ll] = getNC(yi, prob_new)
        resids_new=pd.DataFrame(resids_new)

        if len(nc_new)==0:
            nc_new=resids_new
        else:
            nc_new += resids_new
    nc_new.columns = clas
    nc_new/=N

    p={}
    for c,item in enumerate(clas):
        p[item]=[float(np.sum(nc_new[item][i]<resids_LOO)+np.random.uniform(0,1,1)*(np.sum(nc_new[item][i]==resids_LOO)+1))/(N+1) for i in range(len(X1))]
    p=pd.DataFrame(p)

    pred=[]
    ci=[]
    for i in range(len(X1)):
        cur=p.iloc[i]
        pred.append(cur.idxmax())
        ci.append([cur[cur==a].index[0] for a in cur if a>alpha])
    
    return ci,p    




     


###############################
## regular conformal
################################
def splitConformalClass(X,Y,X1,fit_func,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    clas=np.unique(Y)   
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    model  = fit_func(x_train,y_train)
    prob = pd.DataFrame(model.predict_proba(x_val), columns=clas)
    resids_split = getNC(y_val, prob)

    prob_new = pd.DataFrame(model.predict_proba(X1), columns=clas)
    nc_new={}
    for ll in range(len(clas)):
        yi=[clas[ll]]*len(X1)
        nc_new[ll] = getNC(yi, prob_new)

    nc_new = (pd.DataFrame(nc_new))
    nc_new.columns = clas
    
    p={}
    for c,item in enumerate(clas):
        p[item]=[float(np.sum(nc_new[item][i]<resids_split)+np.random.uniform(0,1,1)*(np.sum(nc_new[item][i]==resids_split)+1))/(N+1) for i in range(len(X1))]
    p=pd.DataFrame(p)

    pred=[]
    ci=[]
    for i in range(len(X1)):
        cur=p.iloc[i]
        pred.append(cur.idxmax())
        ci.append([cur[cur==a].index[0] for a in cur if a>alpha])
    return ci,p    

## cross
def crossConformalClass(X,Y,X1,fit_func,K=5,alpha=0.1):
    N=len(X)
    N1 = len(X1)
    clas=np.unique(Y)   
    kf = KFold(K, shuffle=True)
    
    resids_cv=[]
    nc_new=pd.DataFrame()
   
    
    for train_index, test_index in kf.split(X):
        x_train, x_val = X[train_index], X[test_index]
        y_train,y_val = Y[train_index], Y[test_index]
        model  = fit_func(x_train,y_train)
        prob = pd.DataFrame(model.predict_proba(x_val), columns=clas)
        resids_cv+=list(getNC(y_val, prob))
        prob_new = pd.DataFrame(model.predict_proba(X1), columns=clas)
        resids_new={}
        
        
        for ll in range(len(clas)):
            yi=[clas[ll]]*len(X1)
            resids_new[ll] = getNC(yi, prob_new)
        resids_new = pd.DataFrame(resids_new)
        
        if len(nc_new)==0:
            nc_new=resids_new
        else:
            nc_new += resids_new
    nc_new.columns = clas
    nc_new/=K
    
    
    p={}
    for c,item in enumerate(clas):
        p[item]=[float(np.sum(nc_new[item][i]<resids_cv)+np.random.uniform(0,1,1)*(np.sum(nc_new[item][i]==resids_cv)+1))/(N+1) for i in range(len(X1))]
    p=pd.DataFrame(p)

    pred=[]
    ci=[]
    for i in range(len(X1)):
        cur=p.iloc[i]
        pred.append(cur.idxmax())
        ci.append([cur[cur==a].index[0] for a in cur if a>alpha])
    
    return ci,p    
     

# J+
def jacknifeConformalClass(X,Y,X1,fit_func,alpha=0.1):
    '''
    Using mean aggregation
    '''
    N=len(X)
    N1 = len(X1)
    clas=np.unique(Y)   
    resids_LOO =[]
    nc_new =[]
    
    for i in range(N):
        model = fit_func(np.delete(X,i,0),np.delete(Y,i))
        probs = pd.DataFrame(model.predict_proba(np.vstack((X[i,],X1))), columns=clas)
        resids_LOO.append(getNC([Y[i]], probs.iloc[0]))

        prob_new = probs.iloc[1:].reset_index()
        
        resids_new = {}
        for ll in range(len(clas)):
            yi=[clas[ll]]*len(X1)
            resids_new[ll] = getNC(yi, prob_new)
        resids_new=pd.DataFrame(resids_new)

        if len(nc_new)==0:
            nc_new=resids_new
        else:
            nc_new += resids_new
    nc_new.columns = clas
    nc_new/=N

    p={}
    for c,item in enumerate(clas):
        p[item]=[float(np.sum(nc_new[item][i]<resids_LOO)+np.random.uniform(0,1,1)*(np.sum(nc_new[item][i]==resids_LOO)+1))/(N+1) for i in range(len(X1))]
    p=pd.DataFrame(p)

    pred=[]
    ci=[]
    for i in range(len(X1)):
        cur=p.iloc[i]
        pred.append(cur.idxmax())
        ci.append([cur[cur==a].index[0] for a in cur if a>alpha])
    
    return ci,p    
    

## BOOTSTRAP J+
def generateBootstrapSamplesClass(Y,n, m, B,replace):
    '''
      Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
    '''
    samples_idx={}
    Y_pd=pd.DataFrame(Y.reshape((len(Y),1)))
    for b in range(B):
        #3 stratified sampling

        idx_I =Y_pd.groupby(0, group_keys=False).apply(lambda x: x.sample(frac=m/n))
        idx_I = np.sort(list(idx_I.index)) # stratified sampling of subset of 
#         sample_idx = np.random.choice(n, m,replace=replace)
        samples_idx[b] = idx_I       
    return(samples_idx)




def fitBootstrapModelsClass(X_train, Y_train, X_predict, fit_func, N, n, B,replace):
    '''
      Train B bootstrap estimators and calculate predictions on X_predict
      Return: list of matrices [M,P]
        samples_idx = B-by-n matrix, row b = indices of b-th bootstrap sample
        predictions = B-by-N1 matrix, row b = predictions from b-th bootstrap sample
          (n1=len(X_predict))
    '''
    clas=np.unique(Y_train)
    samples_idx = generateBootstrapSamplesClass(Y_train,N, n, B,replace)
    N1 = len(X_predict)
    # P holds the predictions from individual bootstrap estimators
    predictions = np.zeros((B, N1,len(clas)), dtype=float)
    for b in range(B):
        model  = fit_func(X_train[samples_idx[b], :],Y_train[samples_idx[b], ])
        predictions[b] = pd.DataFrame(model.predict_proba(X_predict), columns=clas)
    return([samples_idx, predictions])


def JplusabClass(X,Y,X1,n_ratio,B,fit_func,replace=False,alpha=0.1):
    '''
    Using mean aggregation
    '''
    N=len(X)
    N1 = len(X1)
    n = np.int(np.round(n_ratio * N))
    clas=np.unique(Y)
    [boot_samples_idx,boot_predictions] = \
        fitBootstrapModelsClass(X, Y, np.vstack((X,X1)), fit_func, N, n, B,replace)
    in_boot_sample = np.zeros((B,N),dtype=bool)
    for b in range(len(in_boot_sample)):
        in_boot_sample[b,boot_samples_idx[b]] = True
    resids_LOO = np.zeros(N)
    
    nc_new =[]
    
    for i in range(N):
        b_keep = np.argwhere(~(in_boot_sample[:,i])).reshape(-1)
        if(len(b_keep)>0):
            probs = pd.DataFrame(boot_predictions[b_keep,i].mean(0).reshape(1,len(clas)), columns=clas)
            resids_LOO[i] = getNC([Y[i]], probs.iloc[0])
            prob_new = pd.DataFrame(boot_predictions[b_keep,N:].mean(0), columns=clas)
            resids_new={}
            for ll in range(len(clas)):
                yi=[clas[ll]]*len(X1)
                resids_new[ll] = getNC(yi, prob_new)
                resids_new=pd.DataFrame(resids_new)

            if len(nc_new)==0:
                nc_new=resids_new
            else:
                nc_new += resids_new
    nc_new.columns = clas
    nc_new/=N
                 
    p={}
    for c,item in enumerate(clas):
        p[item]=[float(np.sum(nc_new[item][i]<resids_LOO)+np.random.uniform(0,1,1)*(np.sum(nc_new[item][i]==resids_LOO)+1))/(N+1) for i in range(len(X1))]
    p=pd.DataFrame(p)

    pred=[]
    ci=[]
    for i in range(len(X1)):
        cur=p.iloc[i]
        pred.append(cur.idxmax())
        ci.append([cur[cur==a].index[0] for a in cur if a>alpha])
    
    return ci,p    