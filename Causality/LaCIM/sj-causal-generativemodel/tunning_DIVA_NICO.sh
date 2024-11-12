#!/bin/bash
for alpha in 1000 2500 10000 25000; do
  for beta in 1000 3500 10000 35000; do
    echo "CUDA_VISIBLE_DEVICES=$1 python $2.py --epochs 200 --optimizer adam --lr $3 --lr_decay 0.2 --lr_controler 60 --in_channel 3 --batch-size 30 --test-batch-size 30 --reg $4 --dataset NICO --data_process fill_std --dataset_type $5 --u_dim 4 --us_dim 1 --num_classes 2 --seed -1 --model DIVA --zs_dim 1024 --env_num $6 --alpha $alpha --beta $beta"
    CUDA_VISIBLE_DEVICES=$1 python $2.py --epochs 200 --optimizer adam --lr $3 --lr_decay 0.2 --lr_controler 60 --in_channel 3 --batch-size 30 --test-batch-size 30 --reg $4 --dataset NICO --data_process fill_std --dataset_type $5 --u_dim 4 --us_dim 1 --num_classes 2 --seed -1 --model DIVA --zs_dim 1024 --env_num $6 --alpha $alpha --beta $beta
  done
done
