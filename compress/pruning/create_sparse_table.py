import glog
import torch
from collections import OrderedDict
from .sparse_ratio_decision import sparse_ratio_decision
import math
import numpy as np
import json
import os

def build_spatial_ratio(layers,init_size,temp_dict,prefix=''):
    keys=layers._modules.keys()
    for i,key in enumerate(keys):
        layer_=layers._modules[key]
        layer_.layer_name=(prefix+'.'+key) if prefix!='' else ''+key
        if isinstance(layer_,torch.nn.Conv2d) or isinstance(layer_,torch.nn.Linear):
            def get_shape(layer,input):
                if len(input[0].shape)==4:
                   layer.spatial_ratio=math.log(init_size//min(input[0].shape[2],input[0].shape[3]))//math.log(2)
                else:
                   layer.spatial_ratio=math.log(init_size)//math.log(2)

                sorted_data = layer.weight.data.abs().view(-1).sort()[0]
                square_data=sorted_data.pow(2)
                energy_all = square_data.sum()
                energy_acc=square_data.cumsum( dim=-1)
                layer.energy_acc=energy_acc
                layer.energy_all=energy_all
                temp_dict[layer.layer_name]=layer
            layer_.get_shape_hook=layer_.register_forward_pre_hook(get_shape)

        if len(layer_._modules.keys())>0:
            build_spatial_ratio(layer_,init_size,temp_dict,prefix=layer_.layer_name)


def parameter_hist(temp_dict,curr_energy,max_spatial_ratio,prun_algo=1.0,prun_algo_tuning=0.5,bin_core_sparse_ratio_decision='core_sparse_ratio_decision',sparse_table={}):
    net_sparse_curr = 0
    net_sparse_max = 0
    MAX_SPARSE=0.99
    total_param=0.0
    total_zero=0.0
    b_achieve_net_max_sparse=False
    glog.info('start analyzing energy threshold {}'.format(curr_energy))
    model_sparse_dict=OrderedDict()
    for layer_name in temp_dict:
        layer=temp_dict[layer_name]
        tmp_energy_th = curr_energy # current energy
        ctrl_bits = 0
        if (isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.Linear) or isinstance(layer,torch.nn.ConvTranspose2d)):

            num_param=layer.weight.numel()
            is_dw=(layer.weight.shape[0]==layer.groups)if not isinstance(layer,torch.nn.Linear)  else 0

            sparse_ratio = sparse_ratio_decision(layer.weight,
                                                layer.energy_acc,
                                                layer.energy_all,
                                                tmp_energy_th,
                                                layer.spatial_ratio,
                                                max_spatial_ratio,
                                                layer.layer_name,
                                                ctrl_bits,
                                                is_dw,
                                                prun_algo,
                                                prun_algo_tuning,
                                                bin_core_sparse_ratio_decision)
            model_sparse_dict[layer_name]=sparse_ratio
            glog.info('pruning ratio for {} is {}'.format(layer_name, sparse_ratio))
            #build_dict(l.name, sparse_ratio, self_sparse_matrix)
            net_sparse_curr += sparse_ratio
            net_sparse_max += MAX_SPARSE if tmp_energy_th else 0
            # STAT
            total_param += num_param
            total_zero += num_param*sparse_ratio

    # STAT
    total_nonzero = total_param - total_zero
    network_sparsification = 1.0*total_zero/total_param

    if (net_sparse_curr>=net_sparse_max) and (curr_energy>0):
        b_achieve_net_max_sparse=True

    return network_sparsification,model_sparse_dict,b_achieve_net_max_sparse

def create_sparse_table(model,input_shape,DEFAULT_SPARSE_STEP=[0.80, 0.82, 0.84, 0.86, 0.88, 0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99],sparse_table_path='sparse_table.json',init_energy=10):
    if os.path.isfile(sparse_table_path):
        with open(sparse_table_path) as f:
            stage_sparse_list=json.load(f)
        return stage_sparse_list
    else:
        # test_index_table = [] # store index for all results from derivation
        # test_sparse_table = [] # store sparsification for all results from derivation
        desire_index_table = [] # store index for desire results from derivation
        desire_sparse_table = [] # store sparsification for desire results from derivation
        # init_energy = 10 # inital energy
        energy_increment = 0
        curr_energy_index = 0
        curr_energy = init_energy
        prev_energy = 0
        curr_sparse = 0
        prev_sparse = 0
        temp_dict=OrderedDict()

        stage_sparse_list=[]
        build_spatial_ratio(model,input_shape[2],temp_dict)
        model(torch.rand(*input_shape).cuda())


        max_spatial_ratio=np.max([temp_dict[key].spatial_ratio for key in temp_dict])
        while len(desire_index_table)<len(DEFAULT_SPARSE_STEP): # to find desired energy step
            # glog.info(len(desire_index_table),len(DEFAULT_SPARSE_STEP))
            # glog.info("current engery {}".format(curr_energy))

            curr_sparse,model_sparse_dict,b_achieve_net_max_sparse = parameter_hist(temp_dict,curr_energy,max_spatial_ratio)

            # test_index_table.append(curr_energy_index)
            # test_sparse_table.append(curr_sparse)
            diff_energy = curr_energy - prev_energy
            diff_sparse = curr_sparse - prev_sparse
            energy_increment = ((DEFAULT_SPARSE_STEP[len(desire_index_table)] - curr_sparse) / (diff_sparse if (diff_sparse!=0) else 0.00001))*diff_energy
            glog.info("{} {} {} {} {} {} {} {} {}".format(curr_energy, curr_sparse, prev_energy, prev_sparse, DEFAULT_SPARSE_STEP[len(desire_index_table)], diff_energy, diff_sparse, DEFAULT_SPARSE_STEP[len(desire_index_table)] - curr_sparse, energy_increment))
            if abs(curr_sparse - DEFAULT_SPARSE_STEP[len(desire_index_table)])<=0.009:
                desire_index_table.append(curr_energy_index)
                desire_sparse_table.append(curr_sparse)
                stage_sparse_list.append([curr_sparse,model_sparse_dict])
            prev_energy = curr_energy
            prev_sparse = curr_sparse
            curr_energy = max(0, curr_energy + energy_increment)
            curr_energy_index = curr_energy_index + 1
            if b_achieve_net_max_sparse:
                print('Analysis is terminated since network already achieves maximum sparsification {} of user''s configuration'.format(curr_sparse))
                break

        for layer_name in temp_dict:
            temp_dict[layer_name].get_shape_hook.remove()
            del temp_dict[layer_name].get_shape_hook
            del temp_dict[layer_name].spatial_ratio
            del temp_dict[layer_name].energy_acc
            del temp_dict[layer_name].energy_all
        with open(sparse_table_path,'w') as f:
            json.dump(stage_sparse_list,f, indent=4)
        return stage_sparse_list

# import torchvision.models as models
# vgg16 = models.vgg16(pretrained=True)
# create_sparse_table(vgg16)