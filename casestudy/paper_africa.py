import sys 
sys.path.append('..')
from LOCO_classification import *
from ML_models import *
from existing_methods_classification import *

import pickle
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
fit_funcs = {
    'DecisionTreeClass':DecisionTreeClass,
    'logitridgecv':logitridgecv,
    'logitridge':logitridge,
    'RFclass':RFclass,
        }

alpha= 0.1
dd = 'africa'
def get_africa():
    dataset = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')
    dataset=dataset.replace({'Present':1,'Absent':0})
    X = np.array(dataset[['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']])

    Y = np.array(dataset['chd'])
    fn=['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']
    return X,Y,fn

X,Y,fn=get_africa()
M=len(X[0])
N=len(X)
X= preprocessing.scale(X)
B=10000
n_ratio = int(np.sqrt(N))/N
m_ratio = 0.5
for fit_func,fit_func_2 in [(DecisionTreeClass,RFclass),(logitridge,logitridgecv)]:
    ress = {}
    print('LOCOMP')
    res = LOCOMPClass(X,Y,n_ratio,m_ratio,B,fit_func, selected_features=[],alpha=0.1,bonf=True)
    ress['LOCO-MP'] = res
    
    print('LOCOSPLIT')
    res_split = LOCOSplitClass(X,Y,fit_func_2,selected_features=[],alpha=0.1,bonf=True)
    ress['LOCO-Split'] = res_split

#     res_vime = vimeeClass(X,Y,fit_func_2,0.001,alpha = 0.1,selected_features=[],measure_type='accuracy')

#     ress['VIMP'] = res_vime
    f = open(dd+'_class_'+fit_func.__name__+'.pkl','wb')
    pickle.dump(ress,f)
    f.close()
