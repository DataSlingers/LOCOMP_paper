library(cpi)
library(mlr)
library(tictoc)
library(mlr3learners)
library(floodgate)
library(methods)
library(glmnet)
library(randomForest)  # to compare forest implementations
library(rpart)
library("reticulate")
source_python('../simulation_regression.py')

args = commandArgs(trailingOnly=TRUE)
generate=(args[1])
N = 1000
M = 200
print(N)
dt=lrn("regr.rpart")
svmm =lrn('regr.svm')## default is kernel RBF
ridgecv = lrn('regr.cv_glmnet',alpha=0)
rff = lrn('regr.ranger')
alpha=0.1
fit_funcs_name = c('Ridge','RF','SVM')
print(N)



## generate simulation data through python code


for (snr in c(0:10)){
    results <-as.data.frame(matrix(nrow=1,ncol=11))
    colnames(results)<-c('pv1','lb','ub','LOCO','Data','type','method','N','SNR','itrial','feature_id')

    print(N)
    print(snr)
    print(generate)



    for (itrial in c(1:50)){
      print(itrial)
      py_run_string(paste0("N=",N))
      py_run_string(paste0("snr=",snr))
      py_run_string(paste0("M=",M))
      py_run_string(paste0("itrial=",itrial))
          if (generate == 'linear'){
            py_run_string("X,Y,X1,Y1= SimuLinear(N,M,10000,snr,seed = 123*itrial+456)")
          }
          if (generate == 'correlated'){
            py_run_string("X,Y,X1,Y1 = SimuCorrelated(N,M,10000,snr,seed = 123*itrial+456)")
          }
              if (generate=='nonlinear'){
                py_run_string("X,Y,X1,Y1 =SimuNonlinear(N,M,10000,snr,seed = 123*itrial+456)")
              }

        Y = scale(py$Y)
        X = py$X
        Y1 = scale(py$Y1)
        X1 = py$X1

####### floodgate

          p=ncol(X)
          Xmodel = "gaussian" # covariate distribution
          Ydist = "gaussian" # conditional model of response
          Xmodel = "laplace" # covariate distribution

          gamma_X.list = replicate(p, rep(0, p-1), simplify = FALSE)
          sigma_X.list = rep(1,p)

          K = 100 # number of null replicates

          nulls.list = sample.gaussian.nulls(X = X, S = as.list(1:p), K = K, gamma_X.list_S = gamma_X.list,
                                             sigma_X.list_S = sigma_X.list)
          split.prop=0.5
          n=nrow(X)
          i1 = sample(1:n, floor(n*split.prop))
          i2 = (1:n)[-i1]
          n1 = length(i1)
          n2 = length(i2)
          alevel=0.1        
          for (namee in fit_funcs_name){
            if (namee=='Ridge'){
              fit_func=funs.list[['ridge']]
            }else if(namee=='RF'){
            fit_func=funs.list[['rf']]
              fit_func$active.fun = function(out){
                  return(list(sort(order(out$importance,decreasing=TRUE))))
              }
              # We hijacked the active set function, take top 20 variables according to
              # the forest important measure

              }else if(namee=='SVM '){
                  fit_func={}
                  fit_func$train.fun = function(x, y){return(svm(x, y))}
                  fit_func$predict.fun = function(out, newx){return(predict(out,newx))}
                  fit_func$active.fun= function(out){
                      return(list(c(1:M)))
                      }

              }
          print(c(namee,snr,itrial))        
            fg.out = floodgate(X, as.matrix(Y), i1, i2, nulls.list = nulls.list,
                               gamma_X.list = gamma_X.list, sigma_X.list = sigma_X.list,
                               Xmodel = Xmodel, fit_func, algo = namee,
                               alevel = alevel, verbose = FALSE)



            inf.out = as.data.frame(fg.out$inf.out)
            inf.out=inf.out[c('P-value','LowConfPt','UpConfPt')]
            inf.out$LOCO ='Floodgate'
            inf.out$Data =generate
            inf.out$type ='Regression'
            inf.out$method =namee
            inf.out$N =N
            inf.out$SNR =snr
            inf.out$itrial =itrial
            inf.out$Feature = unlist(fg.out$S)
            colnames(inf.out)=colnames(results2)        
            results=rbind(results,inf.out)

          write.csv(results,paste0('floodgate_',generate,'_reg_',snr,'_.csv'))        
    }
  }
    }
print('done')

