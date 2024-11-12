#!/bin/bash
for lr in 0.005 0.01 0.05 0.1 0.2; do
  for reg in 0.0 0.00005 0.0001 0.0005 0.001 0.005; do
    #echo "CUDA_VISIBLE_DEVICES=$1 python $2.py --in_channel 3 --test_ep 20 --lr2 0.0005 --reg2 0.005 --sample_num 1 --beta 1 --gamma 1 --batch-size 20 --test-batch-size 20 --epochs $3 --optimizer sgd --lr $lr --lr_decay 0.2 --lr_controler 60 --reg $reg  --fix_mu $4 --fix_var $5 --is_use_u $6 --KLD_type $7 --data_process fill_std --dataset $8 --dataset_type $9 --u_dim ${10} --us_dim 5 --image_size 256 --env_num 2 --fold ${11} --root ${12} --zs_dim ${13} --model ${14} --env_num ${15} --alpha ${16} --is_bn ${17}"
    CUDA_VISIBLE_DEVICES=$1 python DANN_baseline.py --epochs 300 --optimizer sgd --batch-size 256 --test-batch-size 256 --lr_decay 0.2 --lr_controler 120 --fix_mu 0 --fix_var 0 --is_use_u 1 --KLD_type 1 --data_process fill_std --dataset mnist_2 --root ./data/colored_MNIST_0.02_env_2_0_c_2/ --u_dim 4 --us_dim 5 --flip 0 --transpose 0 --image_size 28 --in_channel 3 --fold 1 --alpha 0 --beta 0 --gamma 0.0 --lr2 0.001 --num_classes 2 --zs_dim 1024 --dp 0.0 --smaller_net 1 --lr ${lr} --reg ${reg}
  done
done
