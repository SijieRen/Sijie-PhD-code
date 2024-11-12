for ((i=1;i<=$1;i++)); do

    echo 'CUDA_VISIBLE_DEVICES=$2 python main_baseline_new_data.py --batch-size 16 --class_num 2 --optimizer Mixed --lr $3 --epochs 110 --lr_decay $4 --lr_controler $5 --wd 0.0001 --alpha 1 --beta 0.005 --sequence 06-30/06-30-Pred-curr-Baseline-mono-02 --pred_future 1 --model rn18 --final_tanh 0 --seed -1 --filename std --xlsx_name ../review_results/RN_1.xls'
  CUDA_VISIBLE_DEVICES=$2 python main_baseline_new_data.py --batch-size 16 --class_num 2 --optimizer Mixed --lr $3 --epochs 110 --lr_decay $4 --lr_controler $5 --wd 0.0001 --alpha 1 --beta 0.005 --sequence 06-30/06-30-Pred-curr-Baseline-mono-02 --pred_future 1 --model rn18 --final_tanh 0 --seed -1 --filename std --xlsx_name ../review_results/RN_1.xls

done