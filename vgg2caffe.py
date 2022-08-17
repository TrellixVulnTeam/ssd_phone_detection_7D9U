from caffe_models.model_converter.pytorch2caffe.pytorch2caffe_ import create_model,create_model_weight

import numpy as np

import torch

#from caffe_models.vgg2.phone_nobn import phone_128_vgg_float

from caffe_models.vgg2.phone_128_vgg_float_4 import phone_128_vgg_float
import torch.nn as nn

from cnnc_util import torch_binarize_layers,torch_lowrank_layers,torch_mergebn_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import caffe

# caffe.set_mode_gpu()



net = phone_128_vgg_float().to(device)

net = nn.DataParallel(net)

net = net.module



snapshot_path ='/home/disk/qizhongpei/ssd_pytorch/weights/phone_best.pth'

state_dict=torch.load(snapshot_path,map_location=lambda storage, loc: storage)

net.load_state_dict(state_dict,strict=True)



#------------lowrank---------------------------------------

# torch_lowrank_layers(net.cpu(),percentale=0.9,has_bn=False)

# lowrank_snapshot_path = '/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/uploaded_version/lowrank/phone_128_vgg_float_lowrank_newest1.pth'

# net.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)

#----------------------------------------------------------



#------------binary-----------------------------------------



# binary_snapshot_path="/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/binary/phone_128_vgg_float_binary_best.pth"

# torch_binarize_layers(net.cpu())

# state_dict = torch.load(binary_snapshot_path)



# for n,w in state_dict.items():

#     if 'bias' not in n and 'bn' not in n:

#         w=w-torch.mean(w,(1,2,3)).view(-1,1,1,1)

#         state_dict[n]=w

    # print(n,w)

    # exit(0)

# state_dict=state_dict - torch.mean(state_dict,(1,2,3)).view(-1,1,1,1)

#----------------------------------------------------------



#------------purning--------------------------------------------



# pruning_snapshot_path = '/home/disk/yenanfei/OMS_phone/weights/purning/OMS_phone_128_vgg_purne_0.96_best.pth'

# state_dict = torch.load(pruning_snapshot_path)

# for name in state_dict.keys():



#     if 'weight' in name:

#         mask = name.replace('weight','mask')



#         if mask in state_dict.keys():

#             state_dict[name]=state_dict[name]*state_dict[mask]

# net.load_state_dict(state_dict,strict=False)

#----------------------------------------------------------



# dict={}

# for name, param in net.named_parameters():

    # a=torch.mean(param)

    # if a==0:

    # print(name,a)

#     count = (torch.abs(param.view(-1).cpu())==0).sum().float()

#     if count!=0:

#         dict[name]=(count)

# print(dict,count)

# exit(0)





torch_mergebn_layers(net.cpu())

net.to(device)



# print(net)



# net.eval()



net_input=torch.ones([1,1,128,128]).to(device)



create_model_weight(net,net_input,model_path="test.prototxt",weight_path="test.caffemodel",feature_dir="feature_dir")

# from caffe_models.model_converter.pytorch2caffe.pytorch2caffe_ import compare_caffe_model

# compare_caffe_model("test.prototxt","test.caffemodel",net_input.cpu(),"feature_dir")





# pred_det,_=net(net_input)



# caffe_model=caffe.Net('/home/disk1/yenanfei/DMS_phone/ssd_pytorch/phone_128_vgg_float.prototxt','/home/disk1/yenanfei/DMS_phone/ssd_pytorch/phone_128_vgg_float_purning_0.95.caffemodel',caffe.TEST)

# caffe_model.blobs['data'].data[...] = net_input.cpu()

# # Forward pass.

# detections = caffe_model.forward(['detection_out','mbox_priorbox'])

# # belt_results=detections['detection_out']



# print(pred_det,'=====')

# print(detections,'+++++')

