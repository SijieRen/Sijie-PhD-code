#!/bin/bash
for dparam in 1000 2500 10000 25000; do
  for yparam in 1000 3500 10000 35000; do
    echo "CUDA_VISIBLE_DEVICES=$1 python $2.py --model DIVA --epochs 200 --optimizer adam --lr $3 --lr_decay 0.2 --lr_controler 60 --reg $4   --fix_mu 0 --fix_var 0 --is_use_u 0 --KLD_type 0 --data_process fill_std --dataset AD --dataset_type $5 --u_dim 4 --us_dim 5 --fold 1 --root /home/botong/ --alpha $dparam --beta $yparam"
    CUDA_VISIBLE_DEVICES=$1 python $2.py --model DIVA --epochs 200 --optimizer adam --lr $3 --lr_decay 0.2 --lr_controler 60 --reg $4   --fix_mu 0 --fix_var 0 --is_use_u 0 --KLD_type 0 --data_process fill_std --dataset AD --dataset_type $5 --u_dim 4 --us_dim 5 --fold 1 --root /home/botong/ --alpha $dparam --beta $yparam
  done
done
