-environment
environment.yml

-run

CUDA_VISIBLE_DEVICES= gpu_id
python run.py
-b batchsize (8/16/32)
--lr learning rate (0.0005/0.00005/0.00001)
--epochs the number of epoch (200/300)
--trd training data
--vd valid data
--saved save root
--gd Correlation Matrix among training data
(please refer to [1]Chen Z M, Wei X S, Wang P, et al. Multi-label image recognition with graph convolutional networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 5177-5186.)

if test:
    -e


