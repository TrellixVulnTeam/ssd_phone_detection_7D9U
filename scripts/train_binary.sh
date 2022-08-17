CUDA_VISIBLE_DEVICES=2 /home/disk/tanjing/anaconda3/bin/python train_pro.py --name phone_128_float --iter_size 1 --learning_rate 0.00001 --use_atss True  2>&1 | tee phone_128_lowrank_binary_3_4.log
