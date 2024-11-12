Readme.md

Our method with order_1, order_2, order_3

order_1:

ours

    CUDA_VISIBLE_DEVICES=5 python main_LSTM_order_1_ours.py --lr 0.0001 --lr_decay 0.2 --lr_controler 60 --epochs 120 --lr2 0.02  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 20 --test-batch-size 20 --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --alpha 0.1 --beta 1 --beta1 0 --gamma 0.5 --delta1 0.001 --log-interval 2 --isplus 0 --G_net_type G_net --feature_list 8 24 28 17 10 9 15 33 --sequence order_1_200704 --final_tanh 0 --seed -1 --is_plot 0 --filename std --RNN_hidden 256 --dp 0.5
    
    2020-08-04 07:27:17,953 - AL - INFO - best val acc epoch 66, val acc: 0.7569 test acc:
    2020-08-04 07:27:17,953 - AL - INFO - test_acc at grade 0: 0.6519
    2020-08-04 07:27:17,953 - AL - INFO - test_acc at grade 1: 0.7017
    2020-08-04 07:27:17,953 - AL - INFO - test_acc at grade 2: 0.6961
    2020-08-04 07:27:17,953 - AL - INFO - test_acc at grade 3: 0.7680
    2020-08-04 07:27:17,953 - AL - INFO - test_acc at grade 4: 0.8398
    2020-08-04 07:27:17,953 - AL - INFO - mean test acc: 0.7315
    2020-08-04 07:27:17,953 - AL - INFO - best val auc epoch 66, val auc: 0.8099, test auc:
    2020-08-04 07:27:17,953 - AL - INFO - test_auc at grade 0: 0.6863
    2020-08-04 07:27:17,953 - AL - INFO - test_auc at grade 1: 0.7654
    2020-08-04 07:27:17,953 - AL - INFO - test_auc at grade 2: 0.7629
    2020-08-04 07:27:17,953 - AL - INFO - test_auc at grade 3: 0.8444
    2020-08-04 07:27:17,953 - AL - INFO - test_auc at grade 4: 0.8837
    2020-08-04 07:27:17,954 - AL - INFO - mean test auc: 0.7886
    2020-08-04 07:27:24,455 - AL - INFO - save_results_path: ../results/Ecur_order_1_200704/2020-08-04-00-56-47_Mixed_0.000100_60.000000_size_256_ep_120_0_R_Maculae/

RN18
    
    CUDA_VISIBLE_DEVICES=0 python main_baseline_new_data.py --batch-size 20 --class_num 2 --optimizer Mixed --lr 0.1 --epochs 120 --lr_decay 0.2 --lr_controler 80 --wd 0.0001 --alpha 1 --beta 0.005 --sequence 06-30/06-30-Pred-curr-Baseline-mono-02 --pred_future 1 --model baseline --final_tanh 0 --seed -1 --filename std
    
    2020-07-05 16:33:42,221 - AL - INFO - Utill now the best test acc epoch is : 55,? acc is 0.6950276243093922
    2020-07-05 16:33:42,221 - AL - INFO - Utill now the best test AUC epoch is : 55, AUC is 0.7356978233034572
    2020-07-05 16:33:42,296 - AL - INFO - best val acc epoch 53, val acc: 0.6961 test acc:
    2020-07-05 16:33:42,296 - AL - INFO - test_acc at grade 0: 0.6022
    2020-07-05 16:33:42,296 - AL - INFO - test_acc at grade 1: 0.7072
    2020-07-05 16:33:42,296 - AL - INFO - test_acc at grade 2: 0.5967
    2020-07-05 16:33:42,296 - AL - INFO - test_acc at grade 3: 0.6851
    2020-07-05 16:33:42,296 - AL - INFO - test_acc at grade 4: 0.7348
    2020-07-05 16:33:42,296 - AL - INFO - mean test acc: 0.6652
    2020-07-05 16:33:42,296 - AL - INFO - best val auc epoch 52, val auc: 0.7529, test auc:
    2020-07-05 16:33:42,296 - AL - INFO - test_auc at grade 0: 0.5807
    2020-07-05 16:33:42,296 - AL - INFO - test_auc at grade 1: 0.7102
    2020-07-05 16:33:42,296 - AL - INFO - test_auc at grade 2: 0.6799
    2020-07-05 16:33:42,296 - AL - INFO - test_auc at grade 3: 0.7644
    2020-07-05 16:33:42,296 - AL - INFO - test_auc at grade 4: 0.8054
    2020-07-05 16:33:42,296 - AL - INFO - mean test auc: 0.7081
    2020-07-05 16:33:42,807 - AL - INFO - save_results_path: ../results/Ecur_06-30/06-30-Pred-curr-Baseline-mono-02/2020-07-05-13-34-44_Mixed_0.100000_80.000000_size_256_ep_120_0_R_Maculae/

MMF -> need rerun

    CUDA_VISIBLE_DEVICES=0 python main_baseline_new_data.py --batch-size 20 --class_num 2 --optimizer Mixed --lr 0.1 --epochs 120 --lr_decay 0.2 --lr_controler 80 --wd 0.0001 --alpha 1 --beta 0.005 --sequence 06-30/06-30-Pred-curr-Baseline-mono-02 --pred_future 1 --model MM_F --final_tanh 0 --seed -1 --filename std
    
    

order_2:

ours

tunning params: lr, wd, alpha, beta, gamma, dp

    CUDA_VISIBLE_DEVICES=7 python main_LSTM_order_2_ours.py --lr 0.001 --lr_decay 0.2 --lr_controler 60 --epochs 140 --lr2 0.02  --wd 0.0001 --wd2 0.0001 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence high_order_2_seq --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5

    2020-08-04 16:02:28,534 - AL - INFO - model average pred results
    2020-08-04 16:02:28,534 - AL - INFO - best val acc epoch 38, val acc: 0.7652 test acc:
    2020-08-04 16:02:28,534 - AL - INFO - test_acc at grade 0: 0.7017
    2020-08-04 16:02:28,534 - AL - INFO - test_acc at grade 1: 0.6796
    2020-08-04 16:02:28,534 - AL - INFO - test_acc at grade 2: 0.7514
    2020-08-04 16:02:28,534 - AL - INFO - test_acc at grade 3: 0.8122
    2020-08-04 16:02:28,534 - AL - INFO - mean test acc: 0.7362
    2020-08-04 16:02:28,534 - AL - INFO - best val auc epoch 114, val auc: 0.8297, test auc:
    2020-08-04 16:02:28,534 - AL - INFO - test_auc at grade 0: 0.7803
    2020-08-04 16:02:28,534 - AL - INFO - test_auc at grade 1: 0.7876
    2020-08-04 16:02:28,535 - AL - INFO - test_auc at grade 2: 0.8474
    2020-08-04 16:02:28,535 - AL - INFO - test_auc at grade 3: 0.9020
    2020-08-04 16:02:28,535 - AL - INFO - mean test auc: 0.8293
    2020-08-04 16:02:32,541 - AL - INFO - save_results_path: ../results/Ecur_order_high_order_2_seq/2020-08-04-00-57-12_Mixed_0.001000_60.000000_size_256_ep_140_0_R_Maculae/

BS
   
    CUDA_VISIBLE_DEVICES=1 python main_baseline_MA.py --lr 0.01 --lr_decay 0.2 --lr_controler 30 --epochs 120  --wd 0.0001 --image_size 256 --batch-size 20 --test-batch-size 20 --eye R --center Maculae --save_checkpoint 1 --log-interval 2 --optimizer Mixed --filename std_all --feature_list 24 28 17 8 10 9 15 33 --sequence 200629_data_BS --final_tanh 0 --alpha 1 --delta1 0.001 --model BS --pred_future 1
    
    2020-06-30 04:57:46,696 - AL - INFO - Utill now the best test acc epoch is : 34,? acc is 0.7113259668508287
    2020-06-30 04:57:46,697 - AL - INFO - Utill now the best test AUC epoch is : 27, AUC is 0.7689180537772087
    2020-06-30 04:57:46,761 - AL - INFO - best val acc epoch 47, val acc: 0.7238 test acc:
    2020-06-30 04:57:46,761 - AL - INFO - test_acc at grade 0: 0.6796
    2020-06-30 04:57:46,761 - AL - INFO - test_acc at grade 1: 0.6409
    2020-06-30 04:57:46,761 - AL - INFO - test_acc at grade 2: 0.6464
    2020-06-30 04:57:46,761 - AL - INFO - test_acc at grade 3: 0.7459
    2020-06-30 04:57:46,761 - AL - INFO - mean test acc: 0.6782
    2020-06-30 04:57:46,761 - AL - INFO - best val auc epoch 48, val auc: 0.7940, test auc:
    2020-06-30 04:57:46,761 - AL - INFO - test_auc at grade 0: 0.6983
    2020-06-30 04:57:46,762 - AL - INFO - test_auc at grade 1: 0.7173
    2020-06-30 04:57:46,762 - AL - INFO - test_auc at grade 2: 0.7462
    2020-06-30 04:57:46,762 - AL - INFO - test_auc at grade 3: 0.8489
    2020-06-30 04:57:46,762 - AL - INFO - mean test auc: 0.7527
    2020-06-30 04:57:47,181 - AL - INFO - save_results_path: ../results/Ecur_order_200629_data_BS/2020-06-30-01-32-46_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/
   
MM_F 

    CUDA_VISIBLE_DEVICES=3 python main_baseline_MA.py --lr 0.01 --lr_decay 0.2 --lr_controler 30 --epochs 120  --wd 0.0001 --image_size 256 --batch-size 20 --test-batch-size 20 --eye R --center Maculae --save_checkpoint 1 --log-interval 2 --optimizer Mixed --filename std_all --feature_list 24 28 17 8 10 9 15 33 --sequence 200629_data_MMF --final_tanh 0 --alpha 1 --delta1 0.001 --model MM_F --pred_future 1
    
    2020-06-30 04:56:59,164 - AL - INFO - Utill now the best test acc epoch is : 25,? acc is 0.7541436464088397
    2020-06-30 04:56:59,164 - AL - INFO - Utill now the best test AUC epoch is : 25, AUC is 0.8186299615877081
    2020-06-30 04:56:59,245 - AL - INFO - best val acc epoch 31, val acc: 0.7638 test acc:
    2020-06-30 04:56:59,246 - AL - INFO - test_acc at grade 0: 0.7127
    2020-06-30 04:56:59,246 - AL - INFO - test_acc at grade 1: 0.7017
    2020-06-30 04:56:59,246 - AL - INFO - test_acc at grade 2: 0.7459
    2020-06-30 04:56:59,246 - AL - INFO - test_acc at grade 3: 0.7735
    2020-06-30 04:56:59,246 - AL - INFO - mean test acc: 0.7334
    2020-06-30 04:56:59,246 - AL - INFO - best val auc epoch 27, val auc: 0.8028, test auc:
    2020-06-30 04:56:59,246 - AL - INFO - test_auc at grade 0: 0.7325
    2020-06-30 04:56:59,246 - AL - INFO - test_auc at grade 1: 0.7732
    2020-06-30 04:56:59,246 - AL - INFO - test_auc at grade 2: 0.8050
    2020-06-30 04:56:59,246 - AL - INFO - test_auc at grade 3: 0.8648
    2020-06-30 04:56:59,246 - AL - INFO - mean test auc: 0.7939
    2020-06-30 04:56:59,700 - AL - INFO - save_results_path: ../results/Ecur_order_200629_data_MMF/2020-06-30-01-32-59_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/

    

order_3

    CUDA_VISIBLE_DEVICES=9 python main_LSTM_order_3_ours.py --lr 0.001 --lr_decay 0.2 --lr_controler 60 --epochs 140 --lr2 0.05  --wd 0.0001 --wd2 0.00005 --image_size 256 --batch-size 18 --test-batch-size 18 --eye R --center Maculae --optimizer Mixed --save_checkpoint 1 --wcl 0.01 --log-interval 2 --filename std_all --isplus 0 --G_net_type G_net --feature_list 24 28 17 8 10 9 15 33 --sequence high_order_3_seq --final_tanh 0 --alpha 0 --beta 1.0 --gamma 1 --delta1 0.001 --beta1 1 --dp 0.5
    
    2020-08-04 14:09:40,803 - AL - INFO - best val acc epoch 107, val acc: 0.7864 test acc:
    2020-08-04 14:09:40,803 - AL - INFO - test_acc at grade 0: 0.7403
    2020-08-04 14:09:40,803 - AL - INFO - test_acc at grade 1: 0.7293
    2020-08-04 14:09:40,803 - AL - INFO - test_acc at grade 2: 0.7790
    2020-08-04 14:09:40,803 - AL - INFO - mean test acc: 0.7495
    2020-08-04 14:09:40,803 - AL - INFO - best val auc epoch 108, val auc: 0.8368, test auc:
    2020-08-04 14:09:40,803 - AL - INFO - test_auc at grade 0: 0.7954
    2020-08-04 14:09:40,803 - AL - INFO - test_auc at grade 1: 0.8402
    2020-08-04 14:09:40,803 - AL - INFO - test_auc at grade 2: 0.8905
    2020-08-04 14:09:40,803 - AL - INFO - mean test auc: 0.8420
    2020-08-04 14:09:44,387 - AL - INFO - save_results_path: ../results/Ecur_order_high_order_3_seq/2020-08-04-00-57-17_Mixed_0.001000_60.000000_size_256_ep_140_0_R_Maculae/

BS
    
    CUDA_VISIBLE_DEVICES=5 python main_baseline_MA_3.py --lr 0.01 --lr_decay 0.2 --lr_controler 30 --epochs 120  --wd 0.0001 --image_size 256 --batch-size 20 --test-batch-size 20 --eye R --center Maculae --save_checkpoint 1 --log-interval 2 --optimizer Mixed --filename std_all --feature_list 24 28 17 8 10 9 15 33 --sequence 200629_data_BS --final_tanh 0 --alpha 1 --delta1 0.001 --model BS --pred_future 1
    
    2020-06-30 04:42:02,248 - AL - INFO - Utill now the best test acc epoch is : 14,? acc is 0.7016574585635359
    2020-06-30 04:42:02,248 - AL - INFO - Utill now the best test AUC epoch is : 41, AUC is 0.7662825437473324
    2020-06-30 04:42:02,295 - AL - INFO - best val acc epoch 28, val acc: 0.7311 test acc:
    2020-06-30 04:42:02,296 - AL - INFO - test_acc at grade 0: 0.6133
    2020-06-30 04:42:02,296 - AL - INFO - test_acc at grade 1: 0.6961
    2020-06-30 04:42:02,296 - AL - INFO - test_acc at grade 2: 0.7735
    2020-06-30 04:42:02,296 - AL - INFO - mean test acc: 0.6943
    2020-06-30 04:42:02,296 - AL - INFO - best val auc epoch 24, val auc: 0.8022, test auc:
    2020-06-30 04:42:02,296 - AL - INFO - test_auc at grade 0: 0.7019
    2020-06-30 04:42:02,296 - AL - INFO - test_auc at grade 1: 0.7649
    2020-06-30 04:42:02,296 - AL - INFO - test_auc at grade 2: 0.7910
    2020-06-30 04:42:02,296 - AL - INFO - mean test auc: 0.7526
    2020-06-30 04:42:02,733 - AL - INFO - save_results_path: ../results/Ecur_order_200629_data_BS/2020-06-30-01-33-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/


MM_F
    
    CUDA_VISIBLE_DEVICES=9 python main_baseline_MA_3.py --lr 0.01 --lr_decay 0.2 --lr_controler 30 --epochs 120  --wd 0.0001 --image_size 256 --batch-size 20 --test-batch-size 20 --eye R --center Maculae --save_checkpoint 1 --log-interval 2 --optimizer Mixed --filename std_all --feature_list 24 28 17 8 10 9 15 33 --sequence 200629_data_MMF --final_tanh 0 --alpha 1 --delta1 0.001 --model MM_F --pred_future 1
    
    2020-06-30 04:37:13,830 - AL - INFO - one epoch time is 89.852873
    2020-06-30 04:37:13,831 - AL - INFO - Utill now the best test acc epoch is : 42,? acc is 0.7421731123388581
    2020-06-30 04:37:13,831 - AL - INFO - Utill now the best test AUC epoch is : 93, AUC is 0.8344430217669654
    2020-06-30 04:37:13,880 - AL - INFO - best val acc epoch 31, val acc: 0.7624 test acc:
    2020-06-30 04:37:13,880 - AL - INFO - test_acc at grade 0: 0.6575
    2020-06-30 04:37:13,880 - AL - INFO - test_acc at grade 1: 0.6685
    2020-06-30 04:37:13,880 - AL - INFO - test_acc at grade 2: 0.7017
    2020-06-30 04:37:13,880 - AL - INFO - mean test acc: 0.6759
    2020-06-30 04:37:13,880 - AL - INFO - best val auc epoch 99, val auc: 0.8021, test auc:
    2020-06-30 04:37:13,880 - AL - INFO - test_auc at grade 0: 0.7707
    2020-06-30 04:37:13,880 - AL - INFO - test_auc at grade 1: 0.8268
    2020-06-30 04:37:13,880 - AL - INFO - test_auc at grade 2: 0.8611
    2020-06-30 04:37:13,880 - AL - INFO - mean test auc: 0.8195
    2020-06-30 04:37:14,165 - AL - INFO - save_results_path: ../results/Ecur_order_200629_data_MMF/2020-06-30-01-33-31_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/
