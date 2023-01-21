from functions.LOCO_classification import *
from functions.simulation_classification import *
from functions.ML_models import *
from functions.validation_classification import *



import pickle
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
## classification
fit_funcs = {
    'DecisionTreeClass':DecisionTreeClass,
    'logitridgecv':logitridgecv,
    'logitridge':logitridge,
    'RFclass':RFclass,
    'SVC':kernelSVC,
        }

func_name = sys.argv[1]
generate = sys.argv[2]
size=sys.argv[3]
snr=float(sys.argv[4])


fit_func=fit_funcs[func_name]
if fit_func==DecisionTreeClass:
    fit_func_2=RFclass
if fit_func==logitridge:
    fit_func_2=logitridgecv
if fit_func==kernelSVC:
    fit_func_2=kernelSVC
Ns = [100,500,1000,2000,250,750,1250,1500,1750]
M=200
N=Ns[int(size)]
B=100
alpha= 0.1
n_ratio=int(np.sqrt(N))/N
m_ratio=int(np.sqrt(M))/M

ress={}
ress_split={}
ress_splitmp={}

for itrial in range(100):
    print(itrial)
    if generate=='linear':
        X,Y,X1,Y1 = SimuLinearClass(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='nonlinear':
        X,Y,X1,Y1 = SimuNonlinearClass(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='autoregressive':
        X,Y,X1,Y1 = SimuAutoregressiveClass(N,M,10000,snr,seed = 123*itrial+456)
                                     
    res = validateLOCOMPClass(X,Y,X1,Y1, n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False)
    ress[itrial] = res                                     
    f = open('validate_locomp_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
        
    res_split = validateLOCOSplitClass(X,Y,X1,Y1, fit_func_2, selected_features=[0],alpha=0.1,bonf=False)
    ress_split[itrial] = res_split
    f = open('validate_locosplit_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
                                       
                                       
                                       
    res_splitmp = validateLOCOSplitMPClass(X,Y,X1,Y1, n_ratio,m_ratio,B,fit_func, selected_features=[0],alpha=0.1,bonf=False)
    ress_splitmp[itrial] = res_splitmp
    f = open('validate_splitmp_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_splitmp,f)
    f.close()                                       
     