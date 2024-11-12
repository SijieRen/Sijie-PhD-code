#!/bin/bash
for lr in 0.01 0.02; do
  for reg in 0.00005 0.0001 0.0002 0.0005; do
    echo "CUDA_VISIBLE_DEVICES=$1 python $2.py --batch-size 8 --epochs $3 --optimizer sgd --lr $lr --lr_decay 0.2 --lr_controler 60 --reg $reg  --fix_mu $4 --fix_var $5 --is_use_u $6 --KLD_type $7 --beta 1 --data_process fill_std --dataset $8 --dataset_type $9 --u_dim ${10} --us_dim 5 --image_size 48 --env_num 2 --fold ${11} --root ${12} --zs_dim ${13} --model ${14} --env_num ${15}"
    CUDA_VISIBLE_DEVICES=$1 python $2.py --batch-size 8 --epochs $3 --optimizer sgd --lr $lr --lr_decay 0.2 --lr_controler 60 --reg $reg --fix_mu $4 --fix_var $5 --is_use_u $6 --KLD_type $7 --beta 1 --data_process fill_std --dataset $8 --dataset_type $9 --u_dim ${10} --us_dim 5 --image_size 48 --env_num 2 --fold ${11} --root ${12} --zs_dim ${13} --model ${14} --env_num ${15} --alpha ${16} --beta ${17} --gamma ${18}
  done
done
