export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/disk/yenanfei/ambacaffe/python":"."
# mmconvert -sf caffe  -iw SSD_256x256_lowrank_binary_iter_103000_78.9%_nobn.caffemodel -in lowrank_binary_ori_nobn_compatiable.prototxt -om BSD256 -df pytorch
/home/disk/tanjing/anaconda3/envs/py38/bin/python caffe_models/model_converter/convert_caffe2pytorch.py -m /home/disk/yenanfei/DMS_phone/PhoneDataset_recut/deploy.prototxt -w /home/disk/yenanfei/DMS_phone/PhoneDataset_recut/weight/v1_iter_7500.caffemodel -o mbox_loc mbox_conf_flatten -i 1 128 128 -n phone_128_vgg_float -fd  caffe_models/feature_dir
mv phone_128_vgg_float.py phone_128_vgg_float.pth caffe_models/vgg2
rm -r caffe_models/feature_dir
# /home/disk/tanjing/anaconda3/envs/py38/bin/python caffe_models/get_priorbox.py
