
for generate in linear #nonlinear correlated 
       do
       #nohup Rscript cpi_class.R $generate > cpiclass$generate.log &
       #nohup Rscript cpi_reg.R $generate > cpireg$generate.log &
       nohup Rscript floodgate_reg.R $generate > floodgatereg$generate.log &

       done



