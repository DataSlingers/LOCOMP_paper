import sys 
sys.path.append('..')
from LOCO_regression import *
from ML_models import *
from existing_methods_regression import *



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

alpha= 0.1
dd = 'wine'
def get_wine():
    oz = pd.read_csv('winequality-red.csv',sep=';')
    y = (oz['quality'])
    x = oz.drop(columns='quality')
    fn=x.columns
    return x,y,list(fn)


X,Y,fn=get_wine()
X= preprocessing.scale(X)
Y = preprocessing.scale(np.array(Y))
N = len(X)
M=len(X[0])
B=10000
n_ratio=int(np.sqrt(N))/N
m_ratio=0.5

for fit_func,fit_func_2 in [(DecisionTreeReg,RFreg),(ridge2,ridgecv)]:

    ress={}
    print('LOCOMP')
    res = LOCOMPReg(X,Y,n_ratio,m_ratio,B,fit_func,selected_features=[],alpha=0.1,bonf=False)
    ress['LOCO-MP'] = res
    print('LOCOSPLIT')
    res_split = LOCOSplitReg(X,Y,fit_func_2,selected_features=[],alpha=0.1,bonf=False)
    ress['LOCO-Split'] = res_split
#     print('vime')
#     res_vime = vimee(X,Y,fit_func_2,alpha = 0.1,selected_features=[])

#     ress['VIMP'] = res_vime

#     print('gcm')
#     GCM = gcm(X,Y,fit_func_2,alpha=0.1,selected_features=[])
#     ress['GCM'] = GCM

    f = open(dd+'_reg_'+fit_func.__name__+'.pkl','wb')
    pickle.dump(ress,f)
    f.close()
