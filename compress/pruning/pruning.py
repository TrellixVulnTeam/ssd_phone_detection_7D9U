import glog
from collections import OrderedDict
# from sparse_ratio_decision import sparse_ratio_decision
import numpy as np
import torch
from torch.autograd import Variable

import math
# class Prune:
#     def __init__(self,sparse_coeff,method=None):
#         self.method=method
#         self.sparse_coeff=sparse_coeff
#     def __call__(self,cls):
#         ori_init=cls.__init__
#         ori_repr=cls.__repr__
#         def __init__(layer,*args,**kws):
#             ori_init(layer,*args,**kws)
#             assert hasattr(layer,'weight')
#             if not hasattr(layer,'weight_mask'):
#                 layer.register_buffer('weight_mask', torch.ones_like(layer.weight))
#             with torch.no_grad():
#                 weight_view=torch.abs(layer.weight).view(-1)
#                 n_param=weight_view.size(0)
#                 point=int(np.floor(self.sparse_coeff*n_param))
#                 sort_value,sort_index=weight_view.sort(descending=False)
#                 layer.weight_mask.view(-1)[sort_index[:point]]=0.0
#                 glog.info("PRUN THR: {} RATIO: {}".format(sort_value[sort_index[point-1]],(point*1.0-1)/n_param))
#             def to_prune(layer,input):
#                 with torch.no_grad():
#                     layer.weight.data=layer.weight.data*layer.weight_mask.data
#             layer.register_forward_pre_hook(to_prune)
#             def mask_gradient(layer,grad_input, grad_output):
#                 weight_grad=grad_input[1]
#                 weight_grad.mul_(layer.weight_mask)
#             layer.register_backward_hook(mask_gradient)
#         def __repr__(layer):
#             temp=ori_repr(layer)
#             return "Pruning({})({})".format(self.sparse_coeff,temp)
#         cls.__init__=__init__
#         cls.__repr__=__repr__
#         return cls









def pruning(layers,sparse_table={},prefix=''):
    keys=layers._modules.keys()
    for i,key in enumerate(keys):
        layer_=layers._modules[key]
        layer_.layer_name=(prefix+'.'+key) if prefix!='' else ''+key
        layer_._forward_pre_hooks.clear()
        layer_._backward_hooks.clear()
        layer_.pruned=False
        if isinstance(layer_,torch.nn.Conv2d) or isinstance(layer_,torch.nn.Linear)or isinstance(layer_,torch.nn.ConvTranspose2d):
            layer_.sparse_coeff=sparse_table[layer_.layer_name] if layer_.layer_name in sparse_table else 0.0
            if isinstance(layer_,torch.nn.Linear):
                layer_.is_fc=True
            else:
                layer_.is_fc=False
            if layer_.sparse_coeff>0:
                def to_prune(layer,input):
                    with torch.no_grad():
                        if not hasattr(layer,'mask'):
                            layer.mask=torch.ones_like(layer.weight)
                            if layer.sparse_coeff>0:
                                weight_view=torch.abs(layer.weight).view(-1)
                                n_param=weight_view.size(0)
                                point=int(np.round(layer.sparse_coeff*n_param))
                                sort_value,sort_index=weight_view.sort(descending=False)
                                layer.mask.view(-1)[sort_index[:point]]=0.0
                            
                                glog.info("{} {} PRUN THR: {} RATIO: {}".format(
                                                                                layer.layer_name,
                                                                                layer.__class__.__name__,
                                                                                sort_value[sort_index[point-1]],
                                                                                (point*1.0-1)/n_param)
                                                                                )
                        
                        layer.mask=layer.mask.to(layer.weight.device)
                        layer.weight.data=layer.weight.data*layer.mask
                        layer.pruned=True
                        # glog.info("prune weight")

                def mask_gradient(layer,grad_input, grad_output):
                    if layer.is_fc:
                        weight_grad=grad_input[2]
                        weight_grad_mask=layer.mask.transpose(1,0)
                    else:
                        weight_grad=grad_input[1]
                        weight_grad_mask=layer.mask
                    weight_grad.mul_(weight_grad_mask)
                    # glog.info("mask_gradient")

                layer_._forward_pre_hooks['to_prune']=to_prune
                layer_._backward_hooks['mask_gradient']=mask_gradient
        if len(layer_._modules.keys())>0:
            pruning(layer_,sparse_table=sparse_table,prefix=layer_.layer_name)









