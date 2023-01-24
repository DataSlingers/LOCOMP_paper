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
source_python('../simulation_classification.py')

args = commandArgs(trailingOnly=TRUE)
generate=(args[1])
N = 1000
M = 200


svmm = lrn( 'classif.svm',predict_type = 'prob',kernel = 'radial')## default is kernel RBF
ridgecv = lrn( 'classif.cv_glmnet',alpha=0,predict_type = 'prob')
rff = lrn('classif.ranger',predict_type = 'prob')
alpha=0.1
iters = 100
fit_funcs_name = c('Ridge','RF','SVM')



## generate simulation data through python code


for (snr in c(0:10)){
    print(snr)
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
        py_run_string("X,Y,X1,Y1= SimuLinearClass(N,M,10000,snr,seed = 123*itrial+456)")
      }
      if (generate == 'correlated'){
        py_run_string("X,Y,X1,Y1 = SimuCorrelatedClass(N,M,10000,snr,seed = 123*itrial+456)")
      }
          if (generate=='nonlinear'){
            py_run_string("X,Y,X1,Y1 =SimuNonlinearClass(N,M,10000,snr,seed = 123*itrial+456)")
          }

  ####
  ## cpi
  ####

  Y = (py$Y)
  X = py$X
  train = data.frame(cbind(X))
  train$y = Y
  dd = as_task_classif(x = train, target = "y",id=deparse(substitute(x)),positive='1')

  for (namee in fit_funcs_name){

      if(namee == 'RF'){
          fit_func = rff
      }else if(namee == 'Ridge'){
          fit_func = ridgecv
      }else if(namee =='SVM'){
          fit_func = svmm
      }

      print(c(itrial,namee,snr))
      # tic()
      cpi_lm_log = cpi(task = (dd), learner = fit_func, alpha = 0.1,
                       resampling = rsmp("holdout"),
                       test = "t", measure = 'classif.logloss', log = FALSE)
      cpi_lm_log$lb = cpi_lm_log$CPI-qnorm(1-alpha/2)*cpi_lm_log$SE
      cpi_lm_log$ub = cpi_lm_log$CPI+qnorm(1-alpha/2)*cpi_lm_log$SE
      res=cpi_lm_log[c('p.value','lb','ci.lo')]
      res$LOCO ='CPI'
      res$Data =generate
      res$type ='Classification'
      res$method =namee
      res$N =N
      res$SNR =snr
      res$itrial =itrial
      res$feature=cpi_lm_log$Variable
      print(res)
      print(results)
      colnames(res)=colnames(results)
      results=rbind(results,res)
  }
  write.csv(results,paste0('cpi_',generate,'_class_',snr,'_.csv'))
}

}
print('done')

