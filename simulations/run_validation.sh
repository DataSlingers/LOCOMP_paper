### regression 

for fit in DecisionTreeReg ridge SVR
do
     for generate in linear nonlinear correlated 
        do
               for size in 0 1 2 3 4 5 6 7 8
               do
                       for snr in 0 
                       do
       nohup python3 -u paper_validation_regression.py $fit $generate $size $snr > mp$fit$generate$size$snr.log &
                       done
       done
done
done


## classification 
for fit in DecisionTreeClass logitridge SVC
do
     for generate in linear nonlinear autoregressive
        do
               for size in 0 1 2 3 4 5 6 7 8
               do
                       for snr in 0 
                       do
       nohup python3 -u paper_validation_classification.py $fit $generate $size $snr > mp$fit$generate$size$snr.log &
                       done
       done
done
done

