#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys,os
from pytorch2caffe.graph import build_model_graph,layer_rename
import re
import caffe
import google.protobuf.text_format
import caffe.proto.caffe_pb2 as cp
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf as pb
from collections import OrderedDict
import numpy as np
import torch
from functools import reduce
import glog
import os
class Caffe2PytorchParser:
    def __init__(self,model_name='CaffeModel',outputs=None):
        self.layers_claim_str=""
        self.self_attr_str=""
        self.forward_str=""
        self.top_str=""
        self.model_name=model_name
        self.outputs=outputs
        with open('{}/pytorch_model_template.py'.format(os.path.dirname(__file__))) as f:
            self.template=f.read()
    def add_layer(self,*args,**kargs):
        args=list(map(str,args))
        if len(args)>2:
            kargs_str=''.join(list(map(lambda key:', {}={}'.format(key,kargs[key]),kargs)))
        else:
            kargs_str=', '.join(list(map(lambda key:'{}={}'.format(key,kargs[key]),kargs)))
        self.self_attr_str='self.{}'.format(args[0])
        layer_claim_str='self.{}={}('.format(args[0],args[1])+'{}'.format(', '.join(args[2:]))+'{})\n        '.format(kargs_str)
        self.layers_claim_str=self.layers_claim_str+layer_claim_str
    def connect(self,bottom,top,layer_type):
        self.top_str=','.join(top)
        if layer_type=='InnerProduct':
            self.forward_str=self.forward_str+'{}={}({}.view({}.size(0),-1))\n        '.format(self.top_str,self.self_attr_str,bottom[0],bottom[0])
        else:
            statement_str='{}={}({})\n        '.format(self.top_str,self.self_attr_str,",".join(bottom))
            self.forward_str=self.forward_str+ statement_str if statement_str not in self.forward_str else self.forward_str
    def to_pytorch_model(self,model_path):
        if self.outputs is None:
            self.forward_str=self.forward_str+'return {}'.format(self.top_str)
        else:
            self.forward_str=self.forward_str+'return {}'.format(",".join(self.outputs))
        with open(model_path,'w') as f:
            f.write(self.template.format(self.model_name,self.model_name,self.layers_claim_str,self.forward_str))


def get_netparameter(model):
    with open(model,'r') as f:
        net = cp.NetParameter()
        pb.text_format.Parse(f.read(), net)
        return net

def parse_param(layer):
    layer_type=layer.type
    if layer_type == "Deconvolution":
        return layer.convolution_param
    elif layer_type == "ReLU":
        return layer.relu_param
    else:
        layer_type="_".join(re.findall('[A-Z][a-z]*', layer_type))
        return eval("layer.%s_param"%(layer_type.lower(),))

def default_get(param,attr):
    if attr=="kernel_size":
        return (param.kernel_size[0],param.kernel_size[0]) if len(param.kernel_size)!=0 else (param.kernel_h,param.kernel_w)
    elif attr=="dilation":
        if len(param.dilation)==0:
            return 1
        else:
            return param.dilation[0]
    elif attr=="pad":
        return (param.pad[0],param.pad[0]) if len(param.pad)!=0 else (param.pad_h,param.pad_w)
    elif attr=="stride":
        stride=(param.stride[0],param.stride[0]) if len(param.stride)!=0 else (param.stride_h,param.stride_w)
        if stride[0]==0 and stride[1]==0:
            return(1,1)
        return stride
    elif attr=="shape":
        return param.shape.dim
    else:
        return eval("param.%s"%(attr))


def caffe2pytorch(prototxt_path,weights_path,pytorch_model_path,outputs):
    caffe_model=caffe.Net(prototxt_path,weights_path,caffe.TEST)
    ShapeDict=OrderedDict()
    for top_name in caffe_model.blobs:
        ShapeDict[layer_rename(top_name)]=caffe_model.blobs[top_name].data.shape

    net=get_netparameter(prototxt_path)
    file_name=os.path.basename(pytorch_model_path)
    target_dir=os.path.dirname(pytorch_model_path)
    model_name,ext=os.path.splitext(file_name)
    caffe_name_type_dict=OrderedDict()
    caffe2pytorch_parser=Caffe2PytorchParser(model_name,outputs=outputs)

    seen_bn=False
    bn_name=None
    for i,layer in enumerate(net.layer):
        layer_type=layer.type
        layer_name=layer_rename(layer.name)
        param=parse_param(layer)
        top=tuple(map(layer_rename,layer.top))
        bottom=tuple(map(layer_rename,layer.bottom))
        caffe_name_type_dict[layer_name]=layer_type
        if layer_type == "Convolution":
            kernel_size=default_get(param,'kernel_size')
            padding=default_get(param,'pad')
            stride=default_get(param,'stride')
            bias=default_get(param,'bias_term')
            groups=default_get(param,'group')
            dilation=default_get(param,'dilation')
            in_channels=ShapeDict[bottom[0]][1]
            binary=default_get(param,'binary')
            if binary:
                caffe2pytorch_parser.add_layer(layer_name,
                                'custom_nn.BinaryConv2d',
                                in_channels,
                                param.num_output,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                groups=groups,
                                dilation=dilation,
                                bias=bias)
            else:
                caffe2pytorch_parser.add_layer(layer_name,
                                'nn.Conv2d',
                                in_channels,
                                param.num_output,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                groups=groups,
                                dilation=dilation,
                                bias=bias)
        elif layer_type=='InnerProduct':
            bias=default_get(param,'bias_term')
            in_channels=reduce(lambda x,y:x*y,ShapeDict[bottom[0]][1:])
            caffe2pytorch_parser.add_layer(layer_name,
                                    'nn.Linear',
                                    in_channels,
                                    param.num_output,
                                    bias=bias)
        elif layer_type=="ReLU":
            caffe2pytorch_parser.add_layer(layer_name,
                            'nn.ReLU',
                            inplace=True)
        elif layer_type=="Pooling":
            MAX=0
            AVE=1
            pool=default_get(param,'pool')
            assert pool in [MAX,AVE]
            if pool==MAX:
                kernel_size=param.kernel_size
                stride=param.stride
                caffe2pytorch_parser.add_layer(layer_name,'nn.MaxPool2d',kernel_size=kernel_size,stride=stride)
            else:
                caffe2pytorch_parser.add_layer(layer_name,'nn.AdaptiveAvgPool2d',output_size=ShapeDict[top[0]][2:4])

        elif layer_type=='Interp':
            width=param.width
            height=param.height
            caffe2pytorch_parser.add_layer(layer_name,'nn.Upsample',size=(width,height), mode='"bilinear"', align_corners=True)
        elif layer_type=='Concat':
            axis=default_get(param,'axis')
            caffe2pytorch_parser.add_layer(layer_name,'custom_nn.Concat',axis=axis,n_inputs=len(bottom))
        elif layer_type=='BatchNorm':
            seen_bn=True
            bn_name=layer_name
            in_channels=ShapeDict[bottom[0]][1]
            caffe2pytorch_parser.add_layer(layer_name,'nn.BatchNorm2d',num_features=in_channels)
        elif layer_type=='Permute':
            order=default_get(param,'order')
            caffe2pytorch_parser.add_layer(layer_name,'custom_nn.Permute',order=order)
        elif layer_type=='Flatten':
            caffe2pytorch_parser.add_layer(layer_name,'custom_nn.Flatten')
        elif layer_type=='Reshape':
            shape=default_get(param,'shape')
            caffe2pytorch_parser.add_layer(layer_name,'custom_nn.Reshape',shape=shape)
        elif layer_type=='Softmax':
            axis=default_get(param,'axis')
            caffe2pytorch_parser.add_layer(layer_name,'custom_nn.Softmax',axis=axis)
        elif layer_type=='Scale':
            if seen_bn:
                if caffe_name_type_dict[bn_name]=='BatchNorm':
                    pass
                    seen_bn=False
            else:
                seen_bn=False
                raise NotImplementedError("The bottom layer of Scale layer must be  BatchNorm")
        else:
            raise NotImplementedError(layer_type)


        caffe2pytorch_parser.connect(bottom,top,layer_type)

    caffe2pytorch_parser.to_pytorch_model(pytorch_model_path)
    module = __import__(model_name)
    # exec("from {} import {} as module".format(model_name,model_name))
    model=getattr(module,model_name)()
    state_dict=model.state_dict()
    caffe_param_dict= OrderedDict()
    for key,item in caffe_model.params.items():
        caffe_param_dict[layer_rename(key)]=item

    seen_bn=False
    bn_name=None
    for layer_name in caffe_param_dict:
        layer_type=caffe_name_type_dict[layer_name]
        if layer_type=='Convolution':
            weight=caffe_param_dict[layer_name][0].data
            state_dict['{}.weight'.format(layer_name)].data.copy_(torch.from_numpy(weight))
            bias=caffe_param_dict[layer_name][1].data
            state_dict['{}.bias'.format(layer_name)].data.copy_(torch.from_numpy(bias.flatten()))
            seen_bn=False
        elif layer_type=='InnerProduct':
            weight=caffe_param_dict[layer_name][0].data
            state_dict['{}.weight'.format(layer_name)].data.copy_(torch.from_numpy(weight))
            bias=caffe_param_dict[layer_name][1].data
            state_dict['{}.bias'.format(layer_name)].data.copy_(torch.from_numpy(bias.flatten()))
            seen_bn=False
        elif layer_type=='BatchNorm':
            running_mean=caffe_param_dict[layer_name][0].data/caffe_param_dict[layer_name][2].data
            running_var=caffe_param_dict[layer_name][1].data/caffe_param_dict[layer_name][2].data
            state_dict['{}.running_var'.format(layer_name)].data.copy_(torch.from_numpy(running_var.flatten()))
            state_dict['{}.running_mean'.format(layer_name)].data.copy_(torch.from_numpy(running_mean.flatten()))
            seen_bn=True
            bn_name=layer_name
        elif layer_type=='Scale':
            if seen_bn:
                print(bn_name,layer_name)
                scale_weight=caffe_param_dict[layer_name][0].data
                scale_bias=caffe_param_dict[layer_name][1].data
                state_dict['{}.weight'.format(bn_name)].data.copy_(torch.from_numpy(scale_weight.flatten()))
                state_dict['{}.bias'.format(bn_name)].data.copy_(torch.from_numpy(scale_bias.flatten()))
                seen_bn=False
            else:
                scale_weight=caffe_param_dict[layer_name][0].data
                scale_bias=caffe_param_dict[layer_name][1].data
                state_dict['{}.weight'.format(layer_name)].data.copy_(torch.from_numpy(scale_weight.flatten()))
                state_dict['{}.bias'.format(layer_name)].data.copy_(torch.from_numpy(scale_bias.flatten()))
                seen_bn=False
        elif layer_type=='Normalize':
            pass
        else:
            raise NotImplementedError(layer_type)
    torch.save(state_dict,os.path.join(target_dir,model_name+'.pth'))
    return model



def relu_l2_dist(x,y):
    x=np.clip(x,0,np.inf)
    y=np.clip(y,0,np.inf)
    return np.sqrt(np.sum(np.power(x-y,2)))

def cos_sim(x,y):
    x=x/np.linalg.norm(x,ord=2)
    y=y/np.linalg.norm(y,ord=2)
    return np.sum(x*y)

def relu_cos_sim(x,y):
    x=np.clip(x,0,np.inf)
    y=np.clip(y,0,np.inf)
    x=x/np.linalg.norm(x,ord=2)
    y=y/np.linalg.norm(y,ord=2)
    return np.sum(x*y)

def compare_caffe_pytorch_model(input_numpy,model_path,weight_path,model,feature_dir="feature_dir"):
    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)

    model.eval()
    graph=OrderedDict()
    hooks=[]
    build_model_graph(model,graph,hooks,feature_dir=feature_dir)
    model(torch.from_numpy(input_numpy).unsqueeze(0).float().clone())
    for hook in hooks:
        hook.remove()
    net=caffe.Net(model_path,weight_path,caffe.TEST)
    net.blobs['data'].data[:] =input_numpy
    net.forward()
    feat_files=list(map(lambda x:x[:-4],os.listdir(feature_dir)))
    for top in net.blobs:
        if layer_rename(top) in feat_files:
            caffe_feat=net.blobs[top].data
            pytorch_feat=np.load(os.path.join(feature_dir,"{}.npy".format(layer_rename(top))))

            glog.info("relu_sim:{} sim:{} l2_dist:{} {} {} {}".format(relu_cos_sim(caffe_feat.flatten(),pytorch_feat.flatten()),cos_sim(caffe_feat.flatten(),pytorch_feat.flatten()),relu_l2_dist(caffe_feat.flatten(),pytorch_feat.flatten()),caffe_feat.shape,pytorch_feat.shape,top))



