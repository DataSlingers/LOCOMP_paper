from functions.LOCO_regression import *
from functions.simulation_regression import *
from functions.ML_models import *
from functions.existing_methods_regression import *



import pickle
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
## regression
fit_funcs = {
    'DecisionTreeReg':DecisionTreeReg,
    'ridgecv':ridgecv,
    'ridge':ridge2,
    'RFreg':RFreg,
    'SVR':kernelSVR
        }

func_name = sys.argv[1]
generate = sys.argv[2]
size=sys.argv[3]
snr=float(sys.argv[4])
# snr=1
# size=2
# generate='linear'
# func_name='DecisionTreeReg'

fit_func=fit_funcs[func_name]
if fit_func==DecisionTreeReg:
    fit_func_2=RFreg
if fit_func==ridge2:
    fit_func_2=ridgecv
if fit_func==kernelSVR:
    fit_func_2=kernelSVR
Ns = [100,250,500,750, 1000,2000,1250,1500,1750]
M=200
N=Ns[int(size)]
B=100
alpha= 0.1
n_ratio=int(np.sqrt(N))/N
m_ratio=int(np.sqrt(M))/M

ress={}
ress_split={}
ress_vimy={}
ress_gcm={}
for itrial in range(100):
    print(itrial)
    if generate=='linear':
        X,Y,X1,Y1 = SimuLinear(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='nonlinear':
        X,Y,X1,Y1 = SimuNonlinear(N,M,10000,snr,seed = 123*itrial+456)
    if generate=='autoregressive':
        X,Y,X1,Y1 = SimuAutoregressive(N,M,10000,snr,seed = 123*itrial+456)
                                     
    res = LOCOMPReg(X,Y, n_ratio,m_ratio,B,fit_func,selected_features=[0],alpha=0.1,bonf=False)
    ress[itrial] = res                                     
    f = open('validate_locomp_'+generate+'_reg_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
        
    res_split = LOCOSplitReg(X,Y, fit_func_2, selected_features=[0],alpha=0.1,bonf=False)
    ress_split[itrial] = res_split
    f = open('validate_locosplit_'+generate+'_reg_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_split,f)
    f.close()
      
            
    res_vimy = vimee(X,Y,fit_func_2,alpha = 0.1,selected_features=[0])            
    ress_vimy[itrial] = res_vimy
    f = open('coverage_vimy_'+generate+'_reg_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_vimy,f)
    f.close()


    GCM = gcm(X,Y,fit_func,alpha=0.1,selected_features=[0])
    ress_gcm[itrial] = GCM
    f = open('coverage_gcm_'+generate+'_reg_'+fit_func.__name__+'_'+str(size)+'_'+str(snr)+'_.pkl','wb')
    pickle.dump(ress_gcm,f)
    f.close()
