for ((i=1;i<=$1;i++)); do
	echo "CUDA_VISIBLE_DEVICES=$2 python main_LSTM_order_1_ours.py --alpha $3 --gamma $4 --delta1 $5 --lambda2 $6 --xlsx_name $7 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --beta 1 --beta1 0 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0"
  CUDA_VISIBLE_DEVICES=$2 python main_LSTM_order_1_ours.py --alpha $3 --gamma $4 --delta1 $5 --lambda2 $6 --xlsx_name $7 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --beta 1 --beta1 0 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
done