export CUDA_VISIBLE_DEVICES=1,2
python train_pro.py --name BSD_SSD_256_2 --restore snapshot/BSD_SSD_256.pth --iter_size 1 --learning_rate 0.00001  2>&1 | tee BSD_SSD_256_2.log
