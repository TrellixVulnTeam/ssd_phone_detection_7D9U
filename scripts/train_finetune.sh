CUDA_VISIBLE_DEVICES=0 /home/disk/tanjing/anaconda3/envs/py38/bin/python -W ignore /home/disk/qizhongpei/ssd_pytorch/train_pro.py --name phone_128_vgg_float_3 \
--iter_size 1 --learning_rate 0.001 \
--use_atss true 2>&1 | tee phone_128_vgg.log
# --distillation -- prune --lowrank --binary
