Readme.md

Our method with order_1, order_2, order_3 
with ###add attri###

order_1:

ours

    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_1_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 10-31-modify/order1-ours-003 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
    2020-11-01 05:28:16,229 - AL - INFO - best val acc epoch 98, val acc: 0.7448 test acc:
    2020-11-01 05:28:16,229 - AL - INFO - test_acc at grade 0: 0.6630
    2020-11-01 05:28:16,229 - AL - INFO - test_acc at grade 1: 0.7017
    2020-11-01 05:28:16,229 - AL - INFO - test_acc at grade 2: 0.6740
    2020-11-01 05:28:16,229 - AL - INFO - test_acc at grade 3: 0.7127
    2020-11-01 05:28:16,229 - AL - INFO - test_acc at grade 4: 0.7735
    2020-11-01 05:28:16,229 - AL - INFO - mean test acc: 0.7050
    2020-11-01 05:28:16,229 - AL - INFO - best val auc epoch 114, val auc: 0.7918, test auc:
    2020-11-01 05:28:16,229 - AL - INFO - test_auc at grade 0: 0.7033
    2020-11-01 05:28:16,229 - AL - INFO - test_auc at grade 1: 0.7836
    2020-11-01 05:28:16,229 - AL - INFO - test_auc at grade 2: 0.7689
    2020-11-01 05:28:16,229 - AL - INFO - test_auc at grade 3: 0.8475
    2020-11-01 05:28:16,229 - AL - INFO - test_auc at grade 4: 0.8935
    2020-11-01 05:28:16,229 - AL - INFO - mean test auc: 0.7994
    
    2020-11-01 05:28:27,354 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order1-ours-003/2020-10-31-22-50-13_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
    



    CUDA_VISIBLE_DEVICES=0 python main_baseline_new_data.py --batch-size 20 --class_num 2 --optimizer Mixed --lr 0.1 --epochs 120 --lr_decay 0.2 --lr_controler 80 --wd 0.0001 --alpha 1 --beta 0.005 --sequence 06-30/06-30-Pred-curr-Baseline-mono-02 --pred_future 1 --model MM_F --final_tanh 0 --seed -1 --filename std
    
    ablation study:
    
    LSTM_f_cur
    CUDA_VISIBLE_DEVICES=1 python main_LSTM_order_1_ours_ablation_LSTM_f_cur.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-09-modify/ablation --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
    LSTM_f_res
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_1_ours_ablation_LSTM_res.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-09-modify/ablation --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
   
order_1 add future pred ablation
    CUDA_VISIBLE_DEVICES=4 python main_LSTM_order_1_ours_past_ablation_LSTM_f_cur.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=0 python main_LSTM_order_1_ours_past_ablation_LSTM_f_cur.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-13 08:00:48,596 - AL - INFO - results with gen
        2020-11-13 08:00:48,596 - AL - INFO - best val acc epoch 96, val acc: 0.6420 test acc:
        2020-11-13 08:00:48,596 - AL - INFO - test_acc at grade 0: 0.5856
        2020-11-13 08:00:48,596 - AL - INFO - test_acc at grade 1: 0.6464
        2020-11-13 08:00:48,596 - AL - INFO - test_acc at grade 2: 0.7127
        2020-11-13 08:00:48,597 - AL - INFO - test_acc at grade 3: 0.7403
        2020-11-13 08:00:48,597 - AL - INFO - test_acc at grade 4: 0.7735
        2020-11-13 08:00:48,597 - AL - INFO - mean test acc: 0.6917
        2020-11-13 08:00:48,597 - AL - INFO - best val auc epoch 94, val auc: 0.7580, test auc:
        2020-11-13 08:00:48,597 - AL - INFO - test_auc at grade 0: 0.6015
        2020-11-13 08:00:48,597 - AL - INFO - test_auc at grade 1: 0.7193
        2020-11-13 08:00:48,597 - AL - INFO - test_auc at grade 2: 0.7178
        2020-11-13 08:00:48,597 - AL - INFO - test_auc at grade 3: 0.8277
        2020-11-13 08:00:48,597 - AL - INFO - test_auc at grade 4: 0.8693
        2020-11-13 08:00:48,597 - AL - INFO - mean test auc: 0.7471
        save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-13-01-33-12_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-13 07:49:55,514 - AL - INFO - results with gen
        2020-11-13 07:49:55,514 - AL - INFO - best val acc epoch 106, val acc: 0.6751 test acc:
        2020-11-13 07:49:55,514 - AL - INFO - test_acc at grade 0: 0.6464
        2020-11-13 07:49:55,514 - AL - INFO - test_acc at grade 1: 0.6796
        2020-11-13 07:49:55,514 - AL - INFO - test_acc at grade 2: 0.6685
        2020-11-13 07:49:55,514 - AL - INFO - test_acc at grade 3: 0.7182
        2020-11-13 07:49:55,514 - AL - INFO - test_acc at grade 4: 0.7956
        2020-11-13 07:49:55,514 - AL - INFO - mean test acc: 0.7017
        2020-11-13 07:49:55,514 - AL - INFO - best val auc epoch 107, val auc: 0.7655, test auc:
        2020-11-13 07:49:55,514 - AL - INFO - test_auc at grade 0: 0.6373
        2020-11-13 07:49:55,514 - AL - INFO - test_auc at grade 1: 0.7552
        2020-11-13 07:49:55,514 - AL - INFO - test_auc at grade 2: 0.7364
        2020-11-13 07:49:55,515 - AL - INFO - test_auc at grade 3: 0.8302
        2020-11-13 07:49:55,515 - AL - INFO - test_auc at grade 4: 0.8803
        2020-11-13 07:49:55,515 - AL - INFO - mean test auc: 0.7679
        save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-13-01-15-36_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
           
        
        
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_1_ours_past_ablation_LSTM_res.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_1_ours_past_ablation_LSTM_res.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-13 21:39:18,226 - AL - INFO - results with minus
        2020-11-13 21:39:18,226 - AL - INFO - best val acc epoch 5, val acc: 0.6354 test acc:
        2020-11-13 21:39:18,226 - AL - INFO - test_acc at grade 0: 0.6685
        2020-11-13 21:39:18,226 - AL - INFO - test_acc at grade 1: 0.6685
        2020-11-13 21:39:18,226 - AL - INFO - test_acc at grade 2: 0.6851
        2020-11-13 21:39:18,226 - AL - INFO - test_acc at grade 3: 0.7017
        2020-11-13 21:39:18,227 - AL - INFO - test_acc at grade 4: 0.7403
        2020-11-13 21:39:18,227 - AL - INFO - mean test acc: 0.6928
        2020-11-13 21:39:18,227 - AL - INFO - best val auc epoch 70, val auc: 0.6938, test auc:
        2020-11-13 21:39:18,227 - AL - INFO - test_auc at grade 0: 0.6933
        2020-11-13 21:39:18,227 - AL - INFO - test_auc at grade 1: 0.7010
        2020-11-13 21:39:18,227 - AL - INFO - test_auc at grade 2: 0.7393
        2020-11-13 21:39:18,227 - AL - INFO - test_auc at grade 3: 0.7775
        2020-11-13 21:39:18,227 - AL - INFO - test_auc at grade 4: 0.7982
        2020-11-13 21:39:18,227 - AL - INFO - mean test auc: 0.7419
    
    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
    CUDA_VISIBLE_DEVICES=4 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.00005 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.00005 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-13 22:06:22,762 - AL - INFO - results with average_all
        2020-11-13 22:06:22,762 - AL - INFO - best val acc epoch 86, val acc: 0.7171 test acc:
        2020-11-13 22:06:22,762 - AL - INFO - test_acc at grade 0: 0.6851
        2020-11-13 22:06:22,763 - AL - INFO - test_acc at grade 1: 0.6298
        2020-11-13 22:06:22,763 - AL - INFO - test_acc at grade 2: 0.6519
        2020-11-13 22:06:22,763 - AL - INFO - test_acc at grade 3: 0.7017
        2020-11-13 22:06:22,763 - AL - INFO - test_acc at grade 4: 0.7680
        2020-11-13 22:06:22,763 - AL - INFO - mean test acc: 0.6873
        2020-11-13 22:06:22,763 - AL - INFO - best val auc epoch 117, val auc: 0.7348, test auc:
        2020-11-13 22:06:22,763 - AL - INFO - test_auc at grade 0: 0.6987
        2020-11-13 22:06:22,763 - AL - INFO - test_auc at grade 1: 0.7190
        2020-11-13 22:06:22,763 - AL - INFO - test_auc at grade 2: 0.7323
        2020-11-13 22:06:22,763 - AL - INFO - test_auc at grade 3: 0.7883
        2020-11-13 22:06:22,763 - AL - INFO - test_auc at grade 4: 0.8529
        2020-11-13 22:06:22,763 - AL - INFO - mean test auc: 0.7582
        2020-11-13 22:06:31,364 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-13-16-06-08_Mixed_0.000100_60.000000_size_256_ep_120_0_R_Maculae/

    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.01 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_1_ours_past_ablation_no_LSTM.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0001 --wd 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.01 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
order_1 add future pred
    CUDA_VISIBLE_DEVICES=0 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=1 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-12 18:04:26,880 - AL - INFO - best val acc epoch 82, val acc: 0.7602 test acc:
        2020-11-12 18:04:26,880 - AL - INFO - test_acc at grade 0: 0.6464
        2020-11-12 18:04:26,880 - AL - INFO - test_acc at grade 1: 0.6796
        2020-11-12 18:04:26,881 - AL - INFO - test_acc at grade 2: 0.6961
        2020-11-12 18:04:26,881 - AL - INFO - test_acc at grade 3: 0.7790
        2020-11-12 18:04:26,881 - AL - INFO - test_acc at grade 4: 0.7845
        2020-11-12 18:04:26,881 - AL - INFO - mean test acc: 0.7171
        2020-11-12 18:04:26,881 - AL - INFO - best val auc epoch 106, val auc: 0.7978, test auc:
        2020-11-12 18:04:26,881 - AL - INFO - test_auc at grade 0: 0.7169
        2020-11-12 18:04:26,881 - AL - INFO - test_auc at grade 1: 0.7671
        2020-11-12 18:04:26,881 - AL - INFO - test_auc at grade 2: 0.7974
        2020-11-12 18:04:26,881 - AL - INFO - test_auc at grade 3: 0.8462
        2020-11-12 18:04:26,881 - AL - INFO - test_auc at grade 4: 0.8816
        2020-11-12 18:04:26,881 - AL - INFO - mean test auc: 0.8018
        2020-11-12 18:04:34,642 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-12-10-58-20_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-12 18:15:39,402 - AL - INFO - results with average_all
        2020-11-12 18:15:39,402 - AL - INFO - best val acc epoch 117, val acc: 0.7381 test acc:
        2020-11-12 18:15:39,402 - AL - INFO - test_acc at grade 0: 0.6851
        2020-11-12 18:15:39,402 - AL - INFO - test_acc at grade 1: 0.7182
        2020-11-12 18:15:39,402 - AL - INFO - test_acc at grade 2: 0.7182
        2020-11-12 18:15:39,403 - AL - INFO - test_acc at grade 3: 0.7790
        2020-11-12 18:15:39,403 - AL - INFO - test_acc at grade 4: 0.8177
        2020-11-12 18:15:39,403 - AL - INFO - mean test acc: 0.7436
        2020-11-12 18:15:39,403 - AL - INFO - best val auc epoch 120, val auc: 0.7898, test auc:
        2020-11-12 18:15:39,403 - AL - INFO - test_auc at grade 0: 0.7214
        2020-11-12 18:15:39,403 - AL - INFO - test_auc at grade 1: 0.7736
        2020-11-12 18:15:39,403 - AL - INFO - test_auc at grade 2: 0.7849
        2020-11-12 18:15:39,403 - AL - INFO - test_auc at grade 3: 0.8356
        2020-11-12 18:15:39,403 - AL - INFO - test_auc at grade 4: 0.8738
        2020-11-12 18:15:39,403 - AL - INFO - mean test auc: 0.7978
        2020-11-12 18:15:50,104 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-12-10-58-09_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-13 01:58:31,994 - AL - INFO - best val acc epoch 105, val acc: 0.7370 test acc:
        2020-11-13 01:58:31,994 - AL - INFO - test_acc at grade 0: 0.6685
        2020-11-13 01:58:31,994 - AL - INFO - test_acc at grade 1: 0.6961
        2020-11-13 01:58:31,994 - AL - INFO - test_acc at grade 2: 0.6851
        2020-11-13 01:58:31,994 - AL - INFO - test_acc at grade 3: 0.7569
        2020-11-13 01:58:31,994 - AL - INFO - test_acc at grade 4: 0.7790
        2020-11-13 01:58:31,994 - AL - INFO - mean test acc: 0.7171
        2020-11-13 01:58:31,995 - AL - INFO - best val auc epoch 120, val auc: 0.7760, test auc:
        2020-11-13 01:58:31,995 - AL - INFO - test_auc at grade 0: 0.7328
        2020-11-13 01:58:31,995 - AL - INFO - test_auc at grade 1: 0.7657
        2020-11-13 01:58:31,995 - AL - INFO - test_auc at grade 2: 0.7772
        2020-11-13 01:58:31,995 - AL - INFO - test_auc at grade 3: 0.8239
        2020-11-13 01:58:31,995 - AL - INFO - test_auc at grade 4: 0.8595
        2020-11-13 01:58:31,995 - AL - INFO - mean test auc: 0.7918
        2020-11-13 01:58:38,975 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-12-18-39-22_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae
        /

        
        
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.5 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.5 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0    
    CUDA_VISIBLE_DEVICES=4 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.5 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-12 18:36:25,927 - AL - INFO - results with average_all
        2020-11-12 18:36:25,927 - AL - INFO - best val acc epoch 89, val acc: 0.7492 test acc:
        2020-11-12 18:36:25,927 - AL - INFO - test_acc at grade 0: 0.6851
        2020-11-12 18:36:25,927 - AL - INFO - test_acc at grade 1: 0.7182
        2020-11-12 18:36:25,927 - AL - INFO - test_acc at grade 2: 0.7127
        2020-11-12 18:36:25,927 - AL - INFO - test_acc at grade 3: 0.7569
        2020-11-12 18:36:25,927 - AL - INFO - test_acc at grade 4: 0.7790
        2020-11-12 18:36:25,927 - AL - INFO - mean test acc: 0.7304
        2020-11-12 18:36:25,927 - AL - INFO - best val auc epoch 93, val auc: 0.8002, test auc:
        2020-11-12 18:36:25,927 - AL - INFO - test_auc at grade 0: 0.7273
        2020-11-12 18:36:25,927 - AL - INFO - test_auc at grade 1: 0.7732
        2020-11-12 18:36:25,927 - AL - INFO - test_auc at grade 2: 0.7681
        2020-11-12 18:36:25,927 - AL - INFO - test_auc at grade 3: 0.8350
        2020-11-12 18:36:25,927 - AL - INFO - test_auc at grade 4: 0.8787
        2020-11-12 18:36:25,928 - AL - INFO - mean test auc: 0.7965
    
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.2 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.2 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-12 18:22:55,079 - AL - INFO - results with average_all
        2020-11-12 18:22:55,079 - AL - INFO - best val acc epoch 119, val acc: 0.7392 test acc:
        2020-11-12 18:22:55,079 - AL - INFO - test_acc at grade 0: 0.6906
        2020-11-12 18:22:55,079 - AL - INFO - test_acc at grade 1: 0.7017
        2020-11-12 18:22:55,079 - AL - INFO - test_acc at grade 2: 0.6906
        2020-11-12 18:22:55,079 - AL - INFO - test_acc at grade 3: 0.7459
        2020-11-12 18:22:55,079 - AL - INFO - test_acc at grade 4: 0.7624
        2020-11-12 18:22:55,080 - AL - INFO - mean test acc: 0.7182
        2020-11-12 18:22:55,080 - AL - INFO - best val auc epoch 119, val auc: 0.7824, test auc:
        2020-11-12 18:22:55,080 - AL - INFO - test_auc at grade 0: 0.7388
        2020-11-12 18:22:55,080 - AL - INFO - test_auc at grade 1: 0.7826
        2020-11-12 18:22:55,080 - AL - INFO - test_auc at grade 2: 0.7746
        2020-11-12 18:22:55,080 - AL - INFO - test_auc at grade 3: 0.8286
        2020-11-12 18:22:55,080 - AL - INFO - test_auc at grade 4: 0.8764
        2020-11-12 18:22:55,080 - AL - INFO - mean test auc: 0.8002
        2020-11-12 18:23:02,864 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-12-10-58-49_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
    
    
    ------------------------------- lower has bugs ----------------------------------
    CUDA_VISIBLE_DEVICES=1 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
        2020-11-10 09:04:52,154 - AL - INFO - best val acc epoch 111, val acc: 0.7138 test acc:
        2020-11-10 09:04:52,154 - AL - INFO - test_acc at grade 0: 0.7017
        2020-11-10 09:04:52,154 - AL - INFO - test_acc at grade 1: 0.6685
        2020-11-10 09:04:52,154 - AL - INFO - test_acc at grade 2: 0.6464
        2020-11-10 09:04:52,154 - AL - INFO - test_acc at grade 3: 0.6575
        2020-11-10 09:04:52,154 - AL - INFO - test_acc at grade 4: 0.7624
        2020-11-10 09:04:52,154 - AL - INFO - mean test acc: 0.6873
        2020-11-10 09:04:52,154 - AL - INFO - best val auc epoch 118, val auc: 0.7686, test auc:
        2020-11-10 09:04:52,154 - AL - INFO - test_auc at grade 0: 0.7245
        2020-11-10 09:04:52,154 - AL - INFO - test_auc at grade 1: 0.7641
        2020-11-10 09:04:52,154 - AL - INFO - test_auc at grade 2: 0.7915
        2020-11-10 09:04:52,154 - AL - INFO - test_auc at grade 3: 0.8303
        2020-11-10 09:04:52,154 - AL - INFO - test_auc at grade 4: 0.8603
        2020-11-10 09:04:52,154 - AL - INFO - mean test auc: 0.7942
        2020-11-10 09:04:58,957 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-10-01-43-48_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-11 01:40:38,457 - AL - INFO - results with average_all
        2020-11-11 01:40:38,457 - AL - INFO - best val acc epoch 94, val acc: 0.7193 test acc:
        2020-11-11 01:40:38,457 - AL - INFO - test_acc at grade 0: 0.6188
        2020-11-11 01:40:38,457 - AL - INFO - test_acc at grade 1: 0.6906
        2020-11-11 01:40:38,457 - AL - INFO - test_acc at grade 2: 0.7127
        2020-11-11 01:40:38,457 - AL - INFO - test_acc at grade 3: 0.7790
        2020-11-11 01:40:38,457 - AL - INFO - test_acc at grade 4: 0.7790
        2020-11-11 01:40:38,457 - AL - INFO - mean test acc: 0.7160
        2020-11-11 01:40:38,457 - AL - INFO - best val auc epoch 108, val auc: 0.7922, test auc:
        2020-11-11 01:40:38,457 - AL - INFO - test_auc at grade 0: 0.7010
        2020-11-11 01:40:38,457 - AL - INFO - test_auc at grade 1: 0.7716
        2020-11-11 01:40:38,457 - AL - INFO - test_auc at grade 2: 0.8018
        2020-11-11 01:40:38,457 - AL - INFO - test_auc at grade 3: 0.8533
        2020-11-11 01:40:38,457 - AL - INFO - test_auc at grade 4: 0.8904
        2020-11-11 01:40:38,457 - AL - INFO - mean test auc: 0.8036
        2020-11-11 01:40:45,530 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-10-18-39-23_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        020-11-11 01:36:26,180 - AL - INFO - results with average_all
        2020-11-11 01:36:26,180 - AL - INFO - best val acc epoch 120, val acc: 0.6895 test acc:
        2020-11-11 01:36:26,180 - AL - INFO - test_acc at grade 0: 0.6961
        2020-11-11 01:36:26,180 - AL - INFO - test_acc at grade 1: 0.7017
        2020-11-11 01:36:26,180 - AL - INFO - test_acc at grade 2: 0.7072
        2020-11-11 01:36:26,180 - AL - INFO - test_acc at grade 3: 0.6796
        2020-11-11 01:36:26,180 - AL - INFO - test_acc at grade 4: 0.8011
        2020-11-11 01:36:26,180 - AL - INFO - mean test acc: 0.7171
        2020-11-11 01:36:26,180 - AL - INFO - best val auc epoch 116, val auc: 0.7859, test auc:
        2020-11-11 01:36:26,180 - AL - INFO - test_auc at grade 0: 0.7283
        2020-11-11 01:36:26,180 - AL - INFO - test_auc at grade 1: 0.7682
        2020-11-11 01:36:26,180 - AL - INFO - test_auc at grade 2: 0.7951
        2020-11-11 01:36:26,181 - AL - INFO - test_auc at grade 3: 0.8380
        2020-11-11 01:36:26,181 - AL - INFO - test_auc at grade 4: 0.8703
        2020-11-11 01:36:26,181 - AL - INFO - mean test auc: 0.8000
        2020-11-11 01:36:32,385 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-10-18-39-46_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-11 08:14:38,854 - AL - INFO - results with average_all
        2020-11-11 08:14:38,854 - AL - INFO - best val acc epoch 101, val acc: 0.7050 test acc:
        2020-11-11 08:14:38,854 - AL - INFO - test_acc at grade 0: 0.6354
        2020-11-11 08:14:38,854 - AL - INFO - test_acc at grade 1: 0.6961
        2020-11-11 08:14:38,854 - AL - INFO - test_acc at grade 2: 0.6685
        2020-11-11 08:14:38,854 - AL - INFO - test_acc at grade 3: 0.7901
        2020-11-11 08:14:38,854 - AL - INFO - test_acc at grade 4: 0.8287
        2020-11-11 08:14:38,854 - AL - INFO - mean test acc: 0.7238
        2020-11-11 08:14:38,854 - AL - INFO - best val auc epoch 77, val auc: 0.7918, test auc:
        2020-11-11 08:14:38,854 - AL - INFO - test_auc at grade 0: 0.7262
        2020-11-11 08:14:38,854 - AL - INFO - test_auc at grade 1: 0.7835
        2020-11-11 08:14:38,854 - AL - INFO - test_auc at grade 2: 0.7755
        2020-11-11 08:14:38,854 - AL - INFO - test_auc at grade 3: 0.8484
        2020-11-11 08:14:38,854 - AL - INFO - test_auc at grade 4: 0.8945
        2020-11-11 08:14:38,854 - AL - INFO - mean test auc: 0.8056
        save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-11-01-52-07_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.5 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 1 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
        2020-11-10 09:05:53,003 - AL - INFO - results with average_all
        2020-11-10 09:05:53,003 - AL - INFO - best val acc epoch 108, val acc: 0.7160 test acc:
        2020-11-10 09:05:53,003 - AL - INFO - test_acc at grade 0: 0.6188
        2020-11-10 09:05:53,003 - AL - INFO - test_acc at grade 1: 0.7293
        2020-11-10 09:05:53,003 - AL - INFO - test_acc at grade 2: 0.6575
        2020-11-10 09:05:53,003 - AL - INFO - test_acc at grade 3: 0.7569
        2020-11-10 09:05:53,003 - AL - INFO - test_acc at grade 4: 0.8122
        2020-11-10 09:05:53,003 - AL - INFO - mean test acc: 0.7149
        2020-11-10 09:05:53,003 - AL - INFO - best val auc epoch 104, val auc: 0.7979, test auc:
        2020-11-10 09:05:53,003 - AL - INFO - test_auc at grade 0: 0.6992
        2020-11-10 09:05:53,003 - AL - INFO - test_auc at grade 1: 0.7567
        2020-11-10 09:05:53,003 - AL - INFO - test_auc at grade 2: 0.7881
        2020-11-10 09:05:53,003 - AL - INFO - test_auc at grade 3: 0.8415
        2020-11-10 09:05:53,003 - AL - INFO - test_auc at grade 4: 0.8864
        2020-11-10 09:05:53,003 - AL - INFO - mean test auc: 0.79440000_s��������������������������������������������������������
        2020-11-10 09:06:00,138 - AL - INFO - save_results_path: ../results/Ecur_11-10-modify/order1-ours/2020-11-10-01-44-14_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
    
    CUDA_VISIBLE_DEVICES=0 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 1 --beta1 0.1 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.05  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_1_ours_past.py --alpha 0.1 --beta 0.5 --beta1 0.2 --gamma 0.5 --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence 11-10-modify/order1-ours --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5 --dropout 0
    
order_2:

ours

tunning params: lr, wd, alpha, beta, gamma, dp

    CUDA_VISIBLE_DEVICES=8 python main_LSTM_order_2_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=0 python main_LSTM_order_2_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_2_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=6 python main_LSTM_order_2_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=7 python main_LSTM_order_2_ours.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-02 11:13:28,151 - AL - INFO - best val acc epoch 101, val acc: 0.7707 test acc:
        2020-11-02 11:13:28,151 - AL - INFO - test_acc at grade 0: 0.6906
        2020-11-02 11:13:28,151 - AL - INFO - test_acc at grade 1: 0.7127
        2020-11-02 11:13:28,151 - AL - INFO - test_acc at grade 2: 0.7680
        2020-11-02 11:13:28,151 - AL - INFO - test_acc at grade 3: 0.7790
        2020-11-02 11:13:28,151 - AL - INFO - mean test acc: 0.7376
        2020-11-02 11:13:28,151 - AL - INFO - best val auc epoch 100, val auc: 0.8362, test auc:
        2020-11-02 11:13:28,151 - AL - INFO - test_auc at grade 0: 0.7780
        2020-11-02 11:13:28,151 - AL - INFO - test_auc at grade 1: 0.8074
        2020-11-02 11:13:28,152 - AL - INFO - test_auc at grade 2: 0.8554
        2020-11-02 11:13:28,152 - AL - INFO - test_auc at grade 3: 0.9024
        2020-11-02 11:13:28,152 - AL - INFO - mean test auc: 0.8358
        2020-11-02 11:13:31,315 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-01-22-05-17_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-21 13:21:02,531 - AL - INFO - model average pred results
        2020-11-21 13:21:02,531 - AL - INFO - best val acc epoch 72, val acc: 0.7749 test acc:
        2020-11-21 13:21:02,531 - AL - INFO - test_acc at grade 0: 0.7072
        2020-11-21 13:21:02,531 - AL - INFO - test_acc at grade 1: 0.7293
        2020-11-21 13:21:02,531 - AL - INFO - test_acc at grade 2: 0.7624
        2020-11-21 13:21:02,531 - AL - INFO - test_acc at grade 3: 0.8122
        2020-11-21 13:21:02,531 - AL - INFO - mean test acc: 0.7528
        2020-11-21 13:21:02,531 - AL - INFO - best val auc epoch 52, val auc: 0.8313, test auc:
        2020-11-21 13:21:02,531 - AL - INFO - test_auc at grade 0: 0.7757
        2020-11-21 13:21:02,531 - AL - INFO - test_auc at grade 1: 0.7931
        2020-11-21 13:21:02,532 - AL - INFO - test_auc at grade 2: 0.8526
        2020-11-21 13:21:02,532 - AL - INFO - test_auc at grade 3: 0.9045
        2020-11-21 13:21:02,532 - AL - INFO - mean test auc: 0.8315
        2020-11-21 13:21:06,427 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-21-00-38-52_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-21 14:07:30,503 - AL - INFO - model average pred results
        2020-11-21 14:07:30,503 - AL - INFO - best val acc epoch 24, val acc: 0.7638 test acc:
        2020-11-21 14:07:30,503 - AL - INFO - test_acc at grade 0: 0.7017
        2020-11-21 14:07:30,503 - AL - INFO - test_acc at grade 1: 0.7293
        2020-11-21 14:07:30,503 - AL - INFO - test_acc at grade 2: 0.7624
        2020-11-21 14:07:30,503 - AL - INFO - test_acc at grade 3: 0.8066
        2020-11-21 14:07:30,503 - AL - INFO - mean test acc: 0.7500
        2020-11-21 14:07:30,503 - AL - INFO - best val auc epoch 74, val auc: 0.8197, test auc:
        2020-11-21 14:07:30,503 - AL - INFO - test_auc at grade 0: 0.7818
        2020-11-21 14:07:30,503 - AL - INFO - test_auc at grade 1: 0.8140
        2020-11-21 14:07:30,503 - AL - INFO - test_auc at grade 2: 0.8448
        2020-11-21 14:07:30,503 - AL - INFO - test_auc at grade 3: 0.8949
        2020-11-21 14:07:30,504 - AL - INFO - mean test auc: 0.8339
        2020-11-21 14:07:33,908 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-21-00-39-00_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/

        2020-11-21 13:48:46,894 - AL - INFO - model average pred results
        2020-11-21 13:48:46,894 - AL - INFO - best val acc epoch 75, val acc: 0.7818 test acc:
        2020-11-21 13:48:46,894 - AL - INFO - test_acc at grade 0: 0.7238
        2020-11-21 13:48:46,894 - AL - INFO - test_acc at grade 1: 0.7403
        2020-11-21 13:48:46,894 - AL - INFO - test_acc at grade 2: 0.7735
        2020-11-21 13:48:46,894 - AL - INFO - test_acc at grade 3: 0.8066
        2020-11-21 13:48:46,894 - AL - INFO - mean test acc: 0.7610
        2020-11-21 13:48:46,894 - AL - INFO - best val auc epoch 59, val auc: 0.8342, test auc:
        2020-11-21 13:48:46,894 - AL - INFO - test_auc at grade 0: 0.7876
        2020-11-21 13:48:46,894 - AL - INFO - test_auc at grade 1: 0.7839
        2020-11-21 13:48:46,894 - AL - INFO - test_auc at grade 2: 0.8567
        2020-11-21 13:48:46,894 - AL - INFO - test_auc at grade 3: 0.9022
        2020-11-21 13:48:46,894 - AL - INFO - mean test auc: 0.8326
        2020-11-21 13:48:50,336 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-21-00-39-06_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
    CUDA_VISIBLE_DEVICES=0 python main_LSTM_order_2_ours_ablation_LSTM_f_cur.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_2_ours_ablation_LSTM_f_cur.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-20 11:11:31,955 - AL - INFO - gen pred results
        2020-11-20 11:11:31,955 - AL - INFO - best val acc epoch 118, val acc: 0.7735 test acc:
        2020-11-20 11:11:31,955 - AL - INFO - test_acc at grade 0: 0.7017
        2020-11-20 11:11:31,955 - AL - INFO - test_acc at grade 1: 0.6685
        2020-11-20 11:11:31,955 - AL - INFO - test_acc at grade 2: 0.7293
        2020-11-20 11:11:31,955 - AL - INFO - test_acc at grade 3: 0.7680
        2020-11-20 11:11:31,955 - AL - INFO - mean test acc: 0.7169
        2020-11-20 11:11:31,955 - AL - INFO - best val auc epoch 118, val auc: 0.8303, test auc:
        2020-11-20 11:11:31,955 - AL - INFO - test_auc at grade 0: 0.7656
        2020-11-20 11:11:31,956 - AL - INFO - test_auc at grade 1: 0.7580
        2020-11-20 11:11:31,956 - AL - INFO - test_auc at grade 2: 0.8327
        2020-11-20 11:11:31,956 - AL - INFO - test_auc at grade 3: 0.8599
        2020-11-20 11:11:31,956 - AL - INFO - mean test auc: 0.8040
        save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-19-23-36-06_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-20 13:22:22,366 - AL - INFO - gen pred results
        2020-11-20 13:22:22,366 - AL - INFO - best val acc epoch 119, val acc: 0.7762 test acc:
        2020-11-20 13:22:22,366 - AL - INFO - test_acc at grade 0: 0.7127
        2020-11-20 13:22:22,366 - AL - INFO - test_acc at grade 1: 0.6851
        2020-11-20 13:22:22,366 - AL - INFO - test_acc at grade 2: 0.7293
        2020-11-20 13:22:22,366 - AL - INFO - test_acc at grade 3: 0.7624
        2020-11-20 13:22:22,366 - AL - INFO - mean test acc: 0.7224
        2020-11-20 13:22:22,366 - AL - INFO - best val auc epoch 97, val auc: 0.8320, test auc:
        2020-11-20 13:22:22,366 - AL - INFO - test_auc at grade 0: 0.7601
        2020-11-20 13:22:22,366 - AL - INFO - test_auc at grade 1: 0.7771
        2020-11-20 13:22:22,366 - AL - INFO - test_auc at grade 2: 0.8513
        2020-11-20 13:22:22,366 - AL - INFO - test_auc at grade 3: 0.8835
        2020-11-20 13:22:22,366 - AL - INFO - mean test auc: 0.8180
         ../results/Ecur_10-31-modify/order2-ours-002/2020-11-20-01-03-55_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
    CUDA_VISIBLE_DEVICES=1 python main_LSTM_order_2_ours_ablation_LSTM_res.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_2_ours_ablation_LSTM_res.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=3 python main_LSTM_order_2_ours_ablation_LSTM_res.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=4 python main_LSTM_order_2_ours_ablation_LSTM_res.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-21 06:38:54,639 - AL - INFO - results with res
        2020-11-21 06:38:54,639 - AL - INFO - best val acc epoch 94, val acc: 0.6436 test acc:
        2020-11-21 06:38:54,639 - AL - INFO - test_acc at grade 0: 0.7348
        2020-11-21 06:38:54,639 - AL - INFO - test_acc at grade 1: 0.6906
        2020-11-21 06:38:54,639 - AL - INFO - test_acc at grade 2: 0.7403
        2020-11-21 06:38:54,639 - AL - INFO - test_acc at grade 3: 0.7845
        2020-11-21 06:38:54,639 - AL - INFO - mean test acc: 0.7376
        2020-11-21 06:38:54,639 - AL - INFO - best val auc epoch 60, val auc: 0.7409, test auc:
        2020-11-21 06:38:54,639 - AL - INFO - test_auc at grade 0: 0.7776
        2020-11-21 06:38:54,639 - AL - INFO - test_auc at grade 1: 0.7717
        2020-11-21 06:38:54,639 - AL - INFO - test_auc at grade 2: 0.8401
        2020-11-21 06:38:54,639 - AL - INFO - test_auc at grade 3: 0.8844
        2020-11-21 06:38:54,639 - AL - INFO - mean test auc: 0.8184
        ../results/Ecur_10-31-modify/order2-ours-002/2020-11-20-20-24-50_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-21 06:38:00,545 - AL - INFO - results with res
        2020-11-21 06:38:00,545 - AL - INFO - best val acc epoch 75, val acc: 0.7528 test acc:
        2020-11-21 06:38:00,545 - AL - INFO - test_acc at grade 0: 0.7293
        2020-11-21 06:38:00,545 - AL - INFO - test_acc at grade 1: 0.6685
        2020-11-21 06:38:00,546 - AL - INFO - test_acc at grade 2: 0.7845
        2020-11-21 06:38:00,546 - AL - INFO - test_acc at grade 3: 0.7956
        2020-11-21 06:38:00,546 - AL - INFO - mean test acc: 0.7445
        2020-11-21 06:38:00,546 - AL - INFO - best val auc epoch 61, val auc: 0.7674, test auc:
        2020-11-21 06:38:00,546 - AL - INFO - test_auc at grade 0: 0.7720
        2020-11-21 06:38:00,546 - AL - INFO - test_auc at grade 1: 0.7579
        2020-11-21 06:38:00,546 - AL - INFO - test_auc at grade 2: 0.8609
        2020-11-21 06:38:00,546 - AL - INFO - test_auc at grade 3: 0.8940
        2020-11-21 06:38:00,546 - AL - INFO - mean test auc: 0.8212
        ../results/Ecur_10-31-modify/order2-ours-002/2020-11-20-20-24-56_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/

        2020-11-21 06:59:59,672 - AL - INFO - best val acc epoch 104, val acc: 0.6409 test acc:
        2020-11-21 06:59:59,672 - AL - INFO - test_acc at grade 0: 0.7182
        2020-11-21 06:59:59,672 - AL - INFO - test_acc at grade 1: 0.6630
        2020-11-21 06:59:59,672 - AL - INFO - test_acc at grade 2: 0.7735
        2020-11-21 06:59:59,672 - AL - INFO - test_acc at grade 3: 0.8177
        2020-11-21 06:59:59,672 - AL - INFO - mean test acc: 0.7431
        2020-11-21 06:59:59,672 - AL - INFO - best val auc epoch 92, val auc: 0.6666, test auc:
        2020-11-21 06:59:59,672 - AL - INFO - test_auc at grade 0: 0.7634
        2020-11-21 06:59:59,672 - AL - INFO - test_auc at grade 1: 0.7755
        2020-11-21 06:59:59,672 - AL - INFO - test_auc at grade 2: 0.8414
        2020-11-21 06:59:59,673 - AL - INFO - test_auc at grade 3: 0.8969
        2020-11-21 06:59:59,673 - AL - INFO - mean test auc: 0.8193
        ../results/Ecur_10-31-modify/order2-ours-002/2020-11-20-20-25-03_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
        2020-11-21 06:36:10,405 - AL - INFO - results with res
        2020-11-21 06:36:10,406 - AL - INFO - best val acc epoch 71, val acc: 0.7583 test acc:
        2020-11-21 06:36:10,406 - AL - INFO - test_acc at grade 0: 0.6906
        2020-11-21 06:36:10,406 - AL - INFO - test_acc at grade 1: 0.6575
        2020-11-21 06:36:10,406 - AL - INFO - test_acc at grade 2: 0.7403
        2020-11-21 06:36:10,406 - AL - INFO - test_acc at grade 3: 0.8177
        2020-11-21 06:36:10,406 - AL - INFO - mean test acc: 0.7265
        2020-11-21 06:36:10,406 - AL - INFO - best val auc epoch 59, val auc: 0.7992, test auc:
        2020-11-21 06:36:10,406 - AL - INFO - test_auc at grade 0: 0.7691
        2020-11-21 06:36:10,406 - AL - INFO - test_auc at grade 1: 0.7878
        2020-11-21 06:36:10,406 - AL - INFO - test_auc at grade 2: 0.8360
        2020-11-21 06:36:10,406 - AL - INFO - test_auc at grade 3: 0.8944
        2020-11-21 06:36:10,406 - AL - INFO - mean test auc: 0.8218
        ../results/Ecur_10-31-modify/order2-ours-002/2020-11-20-20-25-11_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
        
    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_2_ours_no_LSTM.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_2_ours_no_LSTM.py --lr 0.0002 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order2-ours-002 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-20 10:34:44,924 - AL - INFO - best val acc epoch 116, val acc: 0.7680 test acc:
        2020-11-20 10:34:44,924 - AL - INFO - test_acc at grade 0: 0.7459
        2020-11-20 10:34:44,924 - AL - INFO - test_acc at grade 1: 0.6961
        2020-11-20 10:34:44,924 - AL - INFO - test_acc at grade 2: 0.7845
        2020-11-20 10:34:44,924 - AL - INFO - test_acc at grade 3: 0.8177
        2020-11-20 10:34:44,925 - AL - INFO - mean test acc: 0.7610
        2020-11-20 10:34:44,925 - AL - INFO - best val auc epoch 116, val auc: 0.8279, test auc:
        2020-11-20 10:34:44,925 - AL - INFO - test_auc at grade 0: 0.7745
        2020-11-20 10:34:44,925 - AL - INFO - test_auc at grade 1: 0.7864
        2020-11-20 10:34:44,925 - AL - INFO - test_auc at grade 2: 0.8449
        2020-11-20 10:34:44,925 - AL - INFO - test_auc at grade 3: 0.9038
        2020-11-20 10:34:44,925 - AL - INFO - mean test auc: 0.8274
        2020-11-20 10:34:48,258 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-19-23-17-02_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/
    
        2020-11-20 10:25:52,025 - AL - INFO - best val acc epoch 20, val acc: 0.7707 test acc:
        2020-11-20 10:25:52,025 - AL - INFO - test_acc at grade 0: 0.6961
        2020-11-20 10:25:52,025 - AL - INFO - test_acc at grade 1: 0.7293
        2020-11-20 10:25:52,025 - AL - INFO - test_acc at grade 2: 0.7901
        2020-11-20 10:25:52,025 - AL - INFO - test_acc at grade 3: 0.7845
        2020-11-20 10:25:52,025 - AL - INFO - mean test acc: 0.7500
        2020-11-20 10:25:52,026 - AL - INFO - best val auc epoch 61, val auc: 0.8303, test auc:
        2020-11-20 10:25:52,026 - AL - INFO - test_auc at grade 0: 0.7864
        2020-11-20 10:25:52,026 - AL - INFO - test_auc at grade 1: 0.8014
        2020-11-20 10:25:52,026 - AL - INFO - test_auc at grade 2: 0.8684
        2020-11-20 10:25:52,026 - AL - INFO - test_auc at grade 3: 0.9106
        2020-11-20 10:25:52,026 - AL - INFO - mean test auc: 0.8417
        2020-11-20 10:25:55,779 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order2-ours-002/2020-11-19-23-26-30_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/



order_3

    CUDA_VISIBLE_DEVICES=2 python main_LSTM_order_3_ours.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    
    2020-11-02 19:49:48,674 - AL - INFO - best val acc epoch 98, val acc: 0.7901 test acc:
    2020-11-02 19:49:48,674 - AL - INFO - test_acc at grade 0: 0.7348
    2020-11-02 19:49:48,674 - AL - INFO - test_acc at grade 1: 0.7624
    2020-11-02 19:49:48,675 - AL - INFO - test_acc at grade 2: 0.7845
    2020-11-02 19:49:48,675 - AL - INFO - mean test acc: 0.7606
    2020-11-02 19:49:48,675 - AL - INFO - best val auc epoch 102, val auc: 0.8473, test auc:
    2020-11-02 19:49:48,675 - AL - INFO - test_auc at grade 0: 0.8192
    2020-11-02 19:49:48,675 - AL - INFO - test_auc at grade 1: 0.8630
    2020-11-02 19:49:48,675 - AL - INFO - test_auc at grade 2: 0.8858
    2020-11-02 19:49:48,675 - AL - INFO - mean test auc: 0.8560
    
    2020-11-02 19:49:52,035 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order3-ours-001/2020-11-02-09-30-16_Mixed_0.000100_60.000000_size_256_ep_110_0_R_Maculae/
    
    CUDA_VISIBLE_DEVICES=6 python main_LSTM_order_3_ours_ablation_LSTM_f_cur.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-20 09:56:59,258 - AL - INFO - gen pred results
        2020-11-20 09:56:59,258 - AL - INFO - best val acc epoch 85, val acc: 0.7808 test acc:
        2020-11-20 09:56:59,258 - AL - INFO - test_acc at grade 0: 0.6796
        2020-11-20 09:56:59,258 - AL - INFO - test_acc at grade 1: 0.7293
        2020-11-20 09:56:59,258 - AL - INFO - test_acc at grade 2: 0.7735
        2020-11-20 09:56:59,258 - AL - INFO - mean test acc: 0.7274
        2020-11-20 09:56:59,258 - AL - INFO - best val auc epoch 65, val auc: 0.8467, test auc:
        2020-11-20 09:56:59,259 - AL - INFO - test_auc at grade 0: 0.7659
        2020-11-20 09:56:59,259 - AL - INFO - test_auc at grade 1: 0.8265
        2020-11-20 09:56:59,259 - AL - INFO - test_auc at grade 2: 0.8682
        2020-11-20 09:56:59,259 - AL - INFO - mean test auc: 0.8202
        save_results_path: ../results/Ecur_10-31-modify/order3-ours-001/2020-11-20-00-02-48_Mixed_0.000100_60.000000_size_256_ep_110_0_R_Maculae/
        
    CUDA_VISIBLE_DEVICES=7 python main_LSTM_order_3_ours_ablation_LSTM_res.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 16 --test-batch-size 16 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-20 08:35:12,709 - AL - INFO - res results
        2020-11-20 08:35:12,709 - AL - INFO - best val acc epoch 1, val acc: 0.6409 test acc:
        2020-11-20 08:35:12,709 - AL - INFO - test_acc at grade 0: 0.6851
        2020-11-20 08:35:12,709 - AL - INFO - test_acc at grade 1: 0.6796
        2020-11-20 08:35:12,709 - AL - INFO - test_acc at grade 2: 0.6409
        2020-11-20 08:35:12,709 - AL - INFO - mean test acc: 0.6685
        2020-11-20 08:35:12,709 - AL - INFO - best val auc epoch 53, val auc: 0.6183, test auc:
        2020-11-20 08:35:12,709 - AL - INFO - test_auc at grade 0: 0.7758
        2020-11-20 08:35:12,709 - AL - INFO - test_auc at grade 1: 0.8134
        2020-11-20 08:35:12,710 - AL - INFO - test_auc at grade 2: 0.8969
        2020-11-20 08:35:12,710 - AL - INFO - mean test auc: 0.8287
        save_results_path: ../results/Ecur_10-31-modify/order3-ours-001/2020-11-20-01-05-38_Mixed_0.000100_60.000000_size_256_ep_110_0_R_Maculae/
        
    CUDA_VISIBLE_DEVICES=8 python main_LSTM_order_3_ours_ablation_no_LSTM.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_3_ours_ablation_no_LSTM.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 110 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence 10-31-modify/order3-ours-001 --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5 --dropout 0
        2020-11-20 10:11:56,101 - AL - INFO - model average pred results
        2020-11-20 10:11:56,101 - AL - INFO - best val acc epoch 80, val acc: 0.7827 test acc:
        2020-11-20 10:11:56,101 - AL - INFO - test_acc at grade 0: 0.7514
        2020-11-20 10:11:56,101 - AL - INFO - test_acc at grade 1: 0.7403
        2020-11-20 10:11:56,101 - AL - INFO - test_acc at grade 2: 0.7901
        2020-11-20 10:11:56,101 - AL - INFO - mean test acc: 0.7606
        2020-11-20 10:11:56,101 - AL - INFO - best val auc epoch 103, val auc: 0.8459, test auc:
        2020-11-20 10:11:56,101 - AL - INFO - test_auc at grade 0: 0.8058
        2020-11-20 10:11:56,101 - AL - INFO - test_auc at grade 1: 0.8470
        2020-11-20 10:11:56,101 - AL - INFO - test_auc at grade 2: 0.8889
        2020-11-20 10:11:56,101 - AL - INFO - mean test auc: 0.8472
        2020-11-20 10:11:59,862 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order3-ours-001/2020-11-20-00-59-45_Mixed_0.000100_60.000000_size_256_ep_110_0_R_Maculae/
        
        2020-11-20 10:12:22,383 - AL - INFO - model average pred results
        2020-11-20 10:12:22,383 - AL - INFO - best val acc epoch 40, val acc: 0.7753 test acc:
        2020-11-20 10:12:22,384 - AL - INFO - test_acc at grade 0: 0.7072
        2020-11-20 10:12:22,384 - AL - INFO - test_acc at grade 1: 0.7680
        2020-11-20 10:12:22,384 - AL - INFO - test_acc at grade 2: 0.7956
        2020-11-20 10:12:22,384 - AL - INFO - mean test acc: 0.7569
        2020-11-20 10:12:22,384 - AL - INFO - best val auc epoch 85, val auc: 0.8469, test auc:
        2020-11-20 10:12:22,384 - AL - INFO - test_auc at grade 0: 0.7990
        2020-11-20 10:12:22,384 - AL - INFO - test_auc at grade 1: 0.8565
        2020-11-20 10:12:22,384 - AL - INFO - test_auc at grade 2: 0.8872
        2020-11-20 10:12:22,384 - AL - INFO - mean test auc: 0.8475
        2020-11-20 10:12:25,901 - AL - INFO - save_results_path: ../results/Ecur_10-31-modify/order3-ours-001/2020-11-20-00-59-53_Mixed_0.000100_60.000000_size_256_ep_110_0_R_Maculae/
