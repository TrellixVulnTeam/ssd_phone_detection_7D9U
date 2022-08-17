import caffe
import numpy as np
from caffe_models.phone_128 import phone_128
from utils.general import xyxy2xywh, xywh2xyxy
net=caffe.Net('caffe_models/deploy.prototxt','caffe_models/v2_iter_100.caffemodel',caffe.TEST)
import torch
torch_param=torch.load('snapshot/phone_128_.pth')
model=phone_128()
def load_student(state_dict):
    from collections import OrderedDict
    new_state_dict=OrderedDict()
    for key in state_dict:
        # print(key)
        # if 'student' in key:
        new_state_dict[key]=state_dict[key]
    return new_state_dict


model.load_state_dict(load_student(torch_param))
print(model)
import cnnc_util
cnnc_util.torch_mergebn_layers(model)
print(model)
torch_param=model.state_dict()
for key,item in torch_param.items():
    print(key)
for key,item in net.params.items():
    if key.endswith('bn') or key.endswith("scale"):
        continue
    key = key.replace('/','_')
    for i in range(len(item)):
        if i==0:
            name='.weight'
        else:
            name='.bias'
        name_id=key+name


        item[i].data[:]=torch_param[name_id].cpu().numpy()[:]

net.save('caffe_models/phone_128.caffemodel')