#!/bin/bash
for lr in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$1 python $2.py --alpha ${18} --beta 1 --gamma 1 --batch-size 8 --epochs $3 --optimizer sgd --lr ${16} --lr_decay 0.2 --lr_controler 60 --reg ${17}  --fix_mu $4 --fix_var $5 --is_use_u $6 --KLD_type $7 --beta 1 --data_process fill_std --dataset $8 --dataset_type $9 --u_dim ${10} --us_dim 5 --image_size 48 --env_num 2 --fold ${11} --root ${12} --zs_dim ${13} --model ${14} --env_num ${15}
done
