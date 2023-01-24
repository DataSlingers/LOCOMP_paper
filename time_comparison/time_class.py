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

import time
generate = 'linear'
N = 500
M=200

B=10000
alpha= 0.1
n_ratio=int(np.sqrt(N))/N
m_ratio=int(np.sqrt(M))/M
results = pd.DataFrame(columns =['N','B','loco','muh_fun','time'])
 
for itrial in range(10):
    for snr in [5]:
        X,Y,X1,Y1 = SimuLinearClass(N,M,10000,snr,seed = 123*itrial+456)
        for fit_func,fit_func_2 in [(DecisionTreeClass,RFclass),(logitridge,logitridgecv)]:
            print('LOCOMP')
            start_time=time.time()
            res = LOCOMPClass(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[],alpha=0.1,bonf=False)
            times= time.time()-start_time
            print(times)
            results.loc[len(results)]=[N,B,'LOCO-MP',fit_func.__name__,times]


            print('LOCOSPLIT')
            start_time=time.time()
            res_split = LOCOSplitClass(X,Y,fit_func_2,selected_features=[],alpha=0.1,bonf=False)
            times= time.time()-start_time
            print(times)

            results.loc[len(results)]=[N,B,'LOCO-Split',fit_func.__name__,times]
            print('vimp')
            start_time=time.time()
            res_vime = vimeeClass(X,Y,fit_func_2,alpha = 0.1,selected_features=[])
            times= time.time()-start_time
            print(times)
            results.loc[len(results)]=[N,B,'VIMP',fit_func.__name__,times]


            results.loc[len(results)]=[N,B,'GCM',fit_func.__name__,times]
        results.to_csv('time_class.csv')

