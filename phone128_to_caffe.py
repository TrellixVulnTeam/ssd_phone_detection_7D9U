# -*- coding: utf-8 -*-

from caffe_models.model_converter.pytorch2caffe.pytorch2caffe import create_model,create_model_weight
import numpy as np
import torch
from caffe_models.phone_128_float import phone_128_float
import torch.nn as nn
from cnnc_util import torch_binarize_layers,torch_lowrank_layers,torch_mergebn_layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = phone_128_float().to(device)
net = nn.DataParallel(net)
net = net.module
net.load_state_dict(torch.load("/home/yenanfei/ssd_pytorch/snapshot/phone_128/phone_128_float.pth", map_location=device))
torch_lowrank_layers(net.cpu(),has_bn=False)
net.to(device)
pruning_snapshot_path = '/home/yenanfei/ssd_pytorch/snapshot/phone_128/phone_128_lowrank_purning_0.90_best.pth'
state_dict = torch.load(pruning_snapshot_path)
net.load_state_dict(state_dict,strict=False)

print(net)
# input()
# torch_mergebn_layers(net)
net.eval()
net_input=torch.ones([1,1,128,128]).to(device)
# numpy_input=np.load("image_input.npy")
# net_input=torch.from_numpy(numpy_input).unsqueeze(0).float()
create_model_weight(net,net_input,model_path="phone_128_lowrank_purning_90%.prototxt",weight_path="phone_128_lowrank_purning_90%.caffemodel",feature_dir="feature_dir")
# from pytorch2caffe import compare_caffe_model
# compare_caffe_model("model_bn.prototxt","model.caffemodel",net_input.cpu(),"feature_dir")
