from collections import OrderedDict
import torch
import sys
sys.path.append('/home/disk/yenanfei/DMS_phone/ssd_pytorch/caffe_models/model_converter/pytorch2caffe/')
from graph import build_model_graph,layer_rename
import mapping as mapping
import numpy as np
import os
import glog
LAYERS=["BatchNorm2d",
        "Conv2d",
        "BinaryConv2d",
        "ConvTranspose2d",
        "Linear",
        "MaxPool2d",
        "Upsample",
        "ReLU",
        "Dropout",
        "L2Norm",
        "Concat",
        "Scale",
        "PReLU",
        "HiQuantConv2d",
        "HiQuantConvTranspose2d",
        "HiQuantLinear",
        "HiQuantMaxPool2d",
        "HiQuantConcat",
        "Flatten",
        "Permute",
        ]
IGNORED_LAYERS=["SoftArgMax","NConv2dX","Conv2dX","Down","Up"]
INPLACE_LAYERS=["BatchNorm2d","ReLU","Dropout","L2Norm","Scale","PReLU"]
PARAM_LAYERS=["BatchNorm2d","Conv2d","ConvTranspose2d","Linear","Scale","L2Norm","HiQuantConv2d","HiQuantConvTranspose2d","HiQuantLinear","BinaryConv2d"]
IGNORED_PARAMS=["bit_width","num_batches_tracked","gfpq_param"]

def relu_l2_dist(x,y):
    x=np.clip(x,0,np.inf)
    y=np.clip(y,0,np.inf)
    return np.sqrt(np.sum(np.power(x-y,2)))

def cos_sim(x,y):
    print('cos_sim')
    x=x/np.linalg.norm(x,ord=2)
    y=y/np.linalg.norm(y,ord=2)
    return np.sum(x*y)

def relu_cos_sim(x,y):
    x=np.clip(x,0,np.inf)
    y=np.clip(y,0,np.inf)
    print("relu_cos_sim")
    x=x/np.linalg.norm(x,ord=2)
    y=y/np.linalg.norm(y,ord=2)
    return np.sum(x*y)

map_trace={}
def inplace_mapping(name):
    name_trace=name
    while(name_trace in map_trace):
        name_trace=map_trace.get(name_trace)
    return name_trace if name_trace is not None else name

def get_last_top(top_names,bottom_name):
    n=len(top_names)
    while(n>=0):
        n=n-1
        if top_names[n].startswith(bottom_name):
            return top_names[n]

def create_model(model,input,device='cuda',model_path="model.prototxt",only_param_layer=False,feature_dir=None):
    graph=OrderedDict()
    hooks_=[]
    if device=="cuda":
        input=input.cuda()
    build_model_graph(model,graph,hooks_,feature_dir=feature_dir)
    print(model)
    model.eval()
    model(input)
    for hook in hooks_:
        hook.remove()
    prototxt_str=""
    unknown_count=0
    unknown_inputs=OrderedDict()
    layers=[]
    top_names=[]
    for layer_name in graph:

        layer=graph[layer_name]
        layer_type=layer["layer_type"]
        # if layer_type =='phone_128_vgg_float':
        #     continue
        # exit(0)
        assert layer_type in LAYERS+IGNORED_LAYERS,layer_type
        config=layer["config"]
        bottom_names=list(map(lambda x:layer_rename(x[0]),layer['config']['bottoms']))
        layer_name=layer_rename(layer_name)
        top_name=layer_name
        ##inplace
        if layer_type in INPLACE_LAYERS:#+IGNORED_LAYERS if not only_param_layer else []:
            map_trace[layer_name]=bottom_names[0]
        top_name=inplace_mapping(top_name)
        bottom_names=list(map(inplace_mapping,bottom_names))
        ##inplace
        if only_param_layer:
            if layer_type in PARAM_LAYERS:
                layers.append(layer_name)
                for i,(bottom,shape) in enumerate(layer['config']['bottoms']):
                    bottom=layer_rename(bottom)
                    if  bottom not in layers:
                        if bottom not in unknown_inputs:
                            unknown_inputs[bottom]=shape
                            prototxt_str=prototxt_str+mapping.create_input(bottom,config,[],bottom,shape=shape)
                prototxt_str=prototxt_str+ eval("mapping.create_{}(layer_name,config,bottom_names,top_name)".format(layer_type.lower()))
        else:
            if layer_type not in IGNORED_LAYERS:
                assert layer_type in LAYERS,layer_type
                top_names.append(top_name)
                for i in range(len(bottom_names)):
                    if bottom_names[i] not in top_names:
                        bottom_names[i]=get_last_top(top_names,bottom_names[i])
                prototxt_str=prototxt_str+ eval("mapping.create_{}(layer_name,config,bottom_names,top_name)".format(layer_type.lower()))
    assert prototxt_str!=""
    with open(model_path,'w') as f:
        f.write(prototxt_str)



def create_model_weight(model,input,model_path="model.prototxt",weight_path="model.caffemodel",feature_dir=None):
    # import sys
    # sys.path.insert(0,'/home/disk/tanjing/projects/adas_pruning/20190718_AmbaCaffe_2.1.6_Autocruis/ambacaffe/python')

    import sys,os
    CAFFE_ROOT = '/home/disk/tanjing/ambacaffe/'
    if os.path.join(CAFFE_ROOT, 'python') not in sys.path:
        sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
    import caffe

    os.system("touch {}".format(weight_path))
    create_model(model,input,device='cuda',model_path=model_path,only_param_layer=True)
    state_dict=model.state_dict()
    state_dict_renamed=OrderedDict()
    for key,item in state_dict.items():
        key_new=key.replace('.','_')
        state_dict_renamed[key_new]=item.cpu().numpy()

    net=caffe.Net(model_path,weight_path,caffe.TEST)
    valid_param_count=0
    for layer_name in net.layer_dict.keys():
        layer=net.layer_dict[layer_name]

        if layer.type in ['Convolution','Deconvolution','InnerProduct']:
            layer.blobs[0].data[:]=state_dict_renamed.pop('{}_weight'.format(layer_name))[:]
            # 二值化
            # layer.blobs[0].data[:] = layer.blobs[0].data[:]- np.mean(layer.blobs[0].data[:],(1,2,3)).reshape(-1,1,1,1)

            valid_param_count=valid_param_count+1
            if '{}_bias'.format(layer_name) in state_dict_renamed:
                layer.blobs[1].data[:]=state_dict_renamed.pop('{}_bias'.format(layer_name))[:]
                valid_param_count=valid_param_count+1
        elif layer.type in ["BatchNorm"]:
            layer.blobs[0].data[:]=state_dict_renamed.pop('{}_running_mean'.format(layer_name))[:]
            layer.blobs[1].data[:]=state_dict_renamed.pop('{}_running_var'.format(layer_name))[:]
            layer.blobs[2].data[:]=np.array(1.0)
            valid_param_count=valid_param_count+3

        elif layer.type in ["Scale"]:
            if layer_name.endswith('_s'):
                layer_name=layer_name[:-2]
            layer.blobs[0].data[:]=state_dict_renamed.pop('{}_weight'.format(layer_name))[:]
            valid_param_count=valid_param_count+1
            if '{}_bias'.format(layer_name) in state_dict_renamed:
                layer.blobs[1].data[:]=state_dict_renamed.pop('{}_bias'.format(layer_name))[:]
                valid_param_count=valid_param_count+1
        elif layer.type in ["Normalize"]:
            layer.blobs[0].data[:]=state_dict_renamed.pop('{}_weight'.format(layer_name))[:]
            valid_param_count=valid_param_count+1
        else:
            glog.info("{} {}".format(layer_name,layer.type))


    for ignored_param in IGNORED_PARAMS:
        params_names=list(state_dict_renamed.keys())
        for param in params_names:
            if param.endswith(ignored_param):
                state_dict_renamed.pop(param)


    assert len(state_dict_renamed)==0,state_dict_renamed
    net.save(weight_path)
    create_model(model,input,device='cuda',model_path=model_path,only_param_layer=False,feature_dir=feature_dir)

def compare_caffe_model(model_path,weight_path,input_numpy,feature_dir):
    import caffe
    net=caffe.Net(model_path,weight_path,caffe.TEST)
    net.blobs['data'].data[:] =input_numpy
    net.forward()
    feat_files=list(map(lambda x:x[:-4],os.listdir(feature_dir)))
    for top in net.blobs:
        if top in feat_files:
            caffe_feat=net.blobs[top].data
            pytorch_feat=np.load(os.path.join(feature_dir,"{}.npy".format(top)))
            caffe_feat=caffe_feat.flatten()
            pytorch_feat=pytorch_feat.flatten()
            glog.info("relu_sim:{} sim:{} l2_dist:{} {} {} {}".format(relu_cos_sim(caffe_feat,pytorch_feat),cos_sim(caffe_feat,pytorch_feat),relu_l2_dist(caffe_feat,pytorch_feat),caffe_feat.shape,pytorch_feat.shape,top))
