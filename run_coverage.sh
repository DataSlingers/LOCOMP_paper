### regression 

for fit in DecisionTreeReg ridge SVR
do
     for generate in linear nonlinear autoregressive
        do
               for size in 1
               do
                       for snr in 0 3 5 7 10 15 
                       do
       nohup python3 -u paper_coverage_regression.py $fit $generate $size $snr > mp$fit$generate$size$snr.log &
                       done
       done
done
done


## classification 
for fit in DecisionTreeClass logitridge SVC
do
     for generate in linear nonlinear autoregressive
        do
               for size in 1
               do
                       for snr in 0 3 5 7 10 15 
                       do
       nohup python3 -u paper_coverage_classification.py $fit $generate $size $snr > mp$fit$generate$size$snr.log &
                       done
       done
done
done

