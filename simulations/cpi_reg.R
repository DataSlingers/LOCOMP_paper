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

dt=lrn( "regr.rpart")
svmm =lrn( 'regr.svm')## default is kernel RBF
ridgecv = lrn( 'regr.cv_glmnet',alpha=0)
rff = lrn('regr.ranger')
alpha=0.1
iters = 100
fit_funcs_name = c('Ridge (CV)','RF','SVM')



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

  ####
  ## cpi
  ####

  Y = py$Y
  X = py$X
  train = data.frame(cbind(X))
  train$y = Y
  dd = as_task_regr(x = train, target = "y")
  for (namee in fit_funcs_name){

      if(namee == 'RF'){
          fit_func = rff
      }else if(namee == 'Ridge (CV)'){
          fit_func = ridgecv
      }else if(namee =='SVM'){
          fit_func = svmm
      }

      print(c(itrial,namee,snr))
      # tic()
        cpi_lm_log = cpi(task = (dd), learner = fit_func, alpha = 0.1,
                         resampling = rsmp("holdout"),
                         test = "t", measure = 'regr.mse', log = FALSE)
        cpi_lm_log$lb = cpi_lm_log$CPI-qnorm(1-alpha/2)*cpi_lm_log$SE
        cpi_lm_log$ub = cpi_lm_log$CPI+qnorm(1-alpha/2)*cpi_lm_log$SE
      
      
      res=cpi_lm_log[c('p.value','lb','ci.lo')]
      res$LOCO ='CPI'
      res$Data =generate
      res$type ='Regression'
      res$method =namee
      res$N =N
      res$SNR =snr
      res$feature=cpi_lm_log$Variable
      colnames(res)=colnames(results)
      results=rbind(results,res)
  }
  write.csv(results,paste0('cpi_',generate,'_reg_',snr,'_.csv'))
}
}
print('done')

