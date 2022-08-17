CUDA_VISIBLE_DEVICES=3 /home/disk/tanjing/anaconda3/bin/python train_pro.py --name phone_128_float \
--iter_size 1 --learning_rate 0.00003 \
3--use_atss True --lowrank --binary 2>&1 | tee phone_128_lowrank_binary_layer_whole_2.log
--use_atss True --lowrank --binary 2>&1 | tee phone_128_lowrank_binary_layer_whole_3.log
# --distillation prune --lowrank --binary
