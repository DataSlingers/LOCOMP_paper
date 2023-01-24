This repository contains code to reproduce results from the paper "Feature Importance Inference with Minipatches]{Model-Agnostic Confidence Intervals for Feature Importance: A Fast and Powerful Approach Using Minipatch Ensembles". 

##  Simulation 
### Figure 1, Figure 2 and Figure 3 
Run LOCOMP and LOCO-Split with SNR = 0 and various N: run paper_validation_regression.py and paper_validation_classification.py by the command “sh simulations/run_validation.sh”



### Figure 4 and Table 1:  
LOCO-MP, LOCO-Split, VIMP, GCM with various SNR: run paper_coverage_regression.py and paper_coverage_classification.py by the command “sh simulations/run_coverage.sh”
CPI and Floodgate: run cpi_reg.R, cpi.class.R, floodgate_reg.R  by the command ‘sh simulations/runR.sh’   

### Table 2
Run Time comparisons  LOCO-MP, LOCO-Split, VIMP, GCM with N = 500, M = 200 by running time_comparison/time_reg.py, time_comparison/time_class.py

Run Time comparisons for CPI and Floodgate by time_comparison/time_reg_r.R, time_comparison/time_class_r.R


## Case Study on Benchmark Data
### Figure 5 
Run LOCO-MP, LOCO-Split by running casestudy/paper_wine.py and casestudy/paper_africa.py 

