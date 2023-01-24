import sys 
sys.path.append('..')

from LOCO_classification import *
from simulation_classification import *
from existing_methods_classification import *
from ML_models import *


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
B=10000
alpha= 0.1
n_ratio=int(np.sqrt(N))/N
m_ratio=int(np.sqrt(M))/M

ress={}
ress_split={}
ress_vimy={}
for itrial in range(100):
    print(itrial)
    if generate=='linear':
        X,Y,X1,Y1 = SimuLinearClass(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='nonlinear':
        X,Y,X1,Y1 = SimuNonlinearClass(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='autoregressive':
        X,Y,X1,Y1 = SimuAutoregressiveClass(N,M,10000,snr,seed = 123*itrial+456)
                                     
    res = LOCOMPClass(X,Y,  n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False)
    ress[itrial] = res                                     
    f = open('coverage_locomp_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
        
    res_split = LOCOSplitClass(X,Y, fit_func_2, selected_features=[0],alpha=0.1,bonf=False)
    ress_split[itrial] = res_split
    f = open('coverage_locosplit_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
                                       
          
            
            
    res_vimy = vimeeClass(X,Y,fit_func_2,alpha = 0.1,selected_features=[0],measure_type='auc')            
    ress_vimy[itrial] = res_vimy
    f = open('coverage_vimy_'+generate+'_class_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_vimy,f)
    f.close()
            