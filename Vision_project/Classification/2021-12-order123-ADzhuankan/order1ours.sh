#!/bin/bash
for alpha in 1 2 3 4; do
    echo "CUDA_VISIBLE_DEVICES=$1 python main_LSTM_order_1_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 1 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0 --lambda2 1 --xlsx_name $2"
    CUDA_VISIBLE_DEVICES=$1 python main_LSTM_order_1_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 1 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0 --lambda2 1 --xlsx_name $2
done