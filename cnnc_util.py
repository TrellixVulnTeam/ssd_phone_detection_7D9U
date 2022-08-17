import sys

# from pytorch_quant import hi_gfpq_fake_quant,hi_gfpq_quant_dequant,KL_Divergence_Quant,HiGFPQ,GFPQ_MODE_INIT,GFPQ_MODE_UPDATE,GFPQ_MODE_APPLY_ONLY,hi_gfpq_quant_dequant_act,hi_gfpq_quant_dequant_weights
import torch.nn as nn
import torch
import torch.nn.functional as F
from  collections import OrderedDict
import numpy as np
import math
import glog
import custom_nn
class ForwardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.mean(torch.abs(w),(1,2,3)).view(-1,1,1,1)*w.sign()


    @staticmethod
    def backward(ctx, g):
        return g

# add binary weight network decorator
def BWN(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)
    assert hasattr(self,'weight')
    params_shape=self.weight.shape
    self.clip_grad=False
    dim=len(params_shape)
    mean_axes=list(range(1,dim))
    reshape_size=[-1]+[1]*(dim-1)
    def hook_bin(layer, inputdata):
        with torch.no_grad():
            layer.weight.org=layer.weight.data.clone()
            sign=torch.sign(layer.weight.org)
            layer.weight.data=torch.reshape(torch.mean(torch.abs(layer.weight.org),mean_axes),reshape_size)*sign



    self._forward_pre_hooks['hook_bin']=hook_bin
    def hook_restore(layer, inputdata, output):
        layer.weight.data.copy_(layer.weight.org)
    self._backward_hooks['hook_restore'] = hook_restore
  param_module_class.__init__=__init__
  return param_module_class


# add binary weight network decorator
def LAB(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    self.optimizer=kws.pop('optimizer')

    orig_init(self,*args,**kws)

    assert hasattr(self,'weight')
    assert hasattr(self,'optimizer')
    params_shape=self.weight.shape
    self.clip_grad=True
    dim=len(params_shape)
    mean_axes=range(1,dim)
    reshape_size=[-1]+[1]*(dim-1)
    self.ori_weight=torch.zeros_like(self.weight)
    def hook_bin(layer, inputdata):
        # print"binarize"
        #layer.weight.data=torch.clamp(layer.weight.data,-1,1)
        with torch.no_grad():
            layer.ori_weight.copy_(layer.weight)
            sign=torch.sign(layer.weight)
            #layer.weight.data=sign
            # print id(layer._parameters['weight'])
            # print id(layer._parameters['weight']) in map(id,layer.optimizer.state)#len(layer.optimizer.state[layer._parameters['weight']])
            # L=layer.optimizer.state[layer.weight]['exp_avg_sq'].sqrt().data+ 1e-8
            #layer.weight.copy_(torch.reshape(torch.mean(torch.abs(layer.weight),mean_axes),reshape_size)*sign)
    self._forward_pre_hooks['hook_bin']=hook_bin
    def hook_restore(layer, inputdata, output):
        # print"restore"
        # if layer.clip_grad:
            # layer.weight.grad[torch.abs(layer.weight.org)>torch.abs(layer.weight.data)]=0
        layer.weight.copy_(layer.ori_weight)

    self._backward_hooks['hook_restore'] = hook_restore

  param_module_class.__init__=__init__
  return param_module_class




# BinaryConv2d=BWN(nn.Conv2d)
# BinaryLinear=BWN(nn.Linear)
# BinaryConvTranspose2d=BWN(nn.ConvTranspose2d)
@BWN
class BinaryConvTranspose2d(nn.ConvTranspose2d):
    pass

class BinaryWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w=w.clone()
        # w[torch.abs(w)<0.001]=0
        # return torch.mean(torch.abs(w),(1,2,3)).view(-1,1,1,1)*w.sign()
        w[torch.abs(w)<0.001]=0
        sign = w.sign()
        sign[sign==0]=1
        return torch.mean(torch.abs(w),(1,2,3)).view(-1,1,1,1)*sign


    @staticmethod
    def backward(ctx, g):
        return g
weight_binarizer = BinaryWeight.apply


############################
class Binary(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        output[output==0]=1
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors

        grad_input = grad_output.clone()
        #****************saturate_ste***************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
 
        return grad_input
binary_fn=Binary.apply
def weight_binarizer_v2(w):
    w=torch.mean(torch.abs(w),(1,2,3)).view(-1,1,1,1)*binary_fn(w)
    return w


class BinaryConv2d(nn.Conv2d):
    def __init__(self,*args,**kws):
        super(BinaryConv2d,self).__init__(*args,**kws)

    def forward(self, input):
        weight = self.weight - torch.mean(self.weight,(1,2,3)).view(-1,1,1,1)
        # weight=weight_binarizer(weight)
        weight=weight_binarizer_v2(weight)

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

                        
################################################# IR-Net
class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class IRConv2d(nn.Conv2d):

    def __init__(self, *args,**kws):
        super(IRConv2d, self).__init__(*args,**kws)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        bw=w
        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # print(bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1))
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = bw.abs().view(bw.size(0), -1).mean(-1).view(bw.size(0), 1, 1, 1)
        # torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).float()).view(bw.size(0), 1, 1, 1).detach()
        # bw.abs().view(bw.size(0), -1).mean(-1).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw
        #print(bw)
        output = F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
############################################################

@BWN
class BinaryLinear(nn.Linear):
  pass


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g
class ClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,min_value,max_value):
        ctx.save_for_backward(x)
        ctx.min_value=min_value
        ctx.max_value=max_value
        return x.clamp(min=min_value,max=max_value)
    @staticmethod
    def backward(ctx, g):
        x,= ctx.saved_tensors
        grad_input = g.clone()
        return grad_input,None,None

round_ste=RoundSTE.apply
clamp_ste=ClampSTE.apply

def gfpq_quant_dequant2(data,max_abs_value,training):
    # max_value=2**(data.abs().max().log2().div(0.0625).floor()).mul(0.0625)
    data_new=data.clone()
    abs_value=data_new.abs_()
    if training:
        max_value=2**(abs_value.detach().max().log2_().div_(0.0625).floor().mul(0.0625))
        if max_value>max_abs_value:
            max_abs_value.data[:]=max_value

    data_new[data_new!=0]=clamp_ste(2**(round_ste(abs_value[data!=0].log2_().div_(0.0625)).mul_(0.0625)),0,max_abs_value.data.item())
    data_new=data.sign()*data_new
    return data_new

class GFPQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data,max_abs_value,log_iterval=0.0625):
        data_new=data.clone()
        data_new[data>0]=2**(data_new[data>0].log2_().div_(log_iterval).round_().mul_(log_iterval))
        data_new[data<0]=-2**((-data_new[data<0]).log2_().div_(log_iterval).round_().mul_(log_iterval))
        data_new[data==0]=0
        ctx.save_for_backward(data,data_new)
        ctx.max_abs_value=max_abs_value
        output=data_new.clamp(-max_abs_value,max_abs_value)
        return output
    @staticmethod
    def backward(ctx, g):
        grad_input = g.clone()
        data,data_new= ctx.saved_tensors
        data_new[data!=0].div_(data[data!=0])
        grad_input[data!=0].mul_(data_new[data!=0])
        grad_input[data<-ctx.max_abs_value]=0
        grad_input[data>ctx.max_abs_value]=0
        return grad_input,None,None

gfpq_ste=GFPQ.apply

def gfpq_quant_dequant3(data,max_abs_value,training,log_iterval=0.0625):
    if training:
        max_value=(2**(data.abs().detach().max().log2_().div_(log_iterval).floor_().mul_(log_iterval))).float()
        if max_value>max_abs_value:
            max_abs_value.data[:]=max_value
    return gfpq_ste(data,max_abs_value.data.item(),log_iterval)



def HI_QUANT2(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)


    self.initized=False
    self.num_gfpq_param=1
    if hasattr(self,'n_inputs'):
        self.num_gfpq_param=self.n_inputs
    if hasattr(self,'groups'):
        self.num_gfpq_param=self.groups
    self.register_buffer('max_abs_value',torch.zeros(1))
    self.register_buffer('bit_width',torch.zeros(1))
    self.bit_width.fill_(8)

    # if hasattr(self,'weight'):
    #     def backward_hook(grad):
    #         grad_input = grad.clone()
    #         self.weight.data[self.weight.org!=0].div_(self.weight.org[self.weight.org!=0])
    #         grad_input[self.weight.org!=0].mul_(self.weight.data[self.weight.org!=0])
    #         return grad_input
    #     self.weight.register_hook(backward_hook)

    def quant_dequant_weight(layer, inputs):
        layer.log_iterval=1.0/16 if layer.bit_width.item()==8 else 1.0/128
        with torch.no_grad():
            if hasattr(layer,'weight'):
                layer.weight.org=layer.weight.data.clone()
                layer.weight.data[layer.weight.org!=0]=2**(layer.weight.data.abs_()[layer.weight.org!=0].log2_().div_(layer.log_iterval).round_().mul_(layer.log_iterval))
                layer.weight.data=layer.weight.org.sign()*layer.weight.data
        return tuple(map(lambda i:gfpq_quant_dequant3(inputs[i],layer.max_abs_value,layer.training,log_iterval=layer.log_iterval),range(len(inputs))))

    self._forward_pre_hooks['quant_dequant_weight']=quant_dequant_weight

    def hook_restore(layer, inputdata, output):
        if hasattr(layer,'weight'):
            layer.weight.data.copy_(layer.weight.org)

    self._backward_hooks['hook_restore'] = hook_restore

    # def quant_dequant_act(layer,inputs,output):
    #     return gfpq_quant_dequant2(output,layer.max_abs_value,layer.training)
    # self._forward_hooks['quant_dequant_act'] = quant_dequant_act

  param_module_class.__init__=__init__
  orig_repr=param_module_class.__repr__
  def __repr__(self):
    return orig_repr(self)+"(bit_width={})".format(self.bit_width.item())
  param_module_class.__repr__=__repr__
  param_module_class.__str__=__repr__

  return param_module_class


def build_parents_graph(layers,layer_dict):
    if "model" not in layer_dict:
        layer_dict["model"]=layers
        layers.layer_name="model"
    keys=layers._modules.keys()
    for i,key in enumerate(keys):
        layer=layers._modules[key]
        if hasattr(layers,'layer_name'):
            layer.parent_name=layers.layer_name
        else:
            layer.parent_name=""
        layer.layer_name=(layer.parent_name+'.'+key) if layer.parent_name!='' else ''+key
        layer_dict[layer.layer_name]=layer
        if hasattr(layer,'_modules'):
            build_parents_graph(layer,layer_dict)

import glog

def HI_QUANT(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)
    self.initized=False
    self.num_gfpq_param=1
    if hasattr(self,'n_inputs'):
        self.num_gfpq_param=self.n_inputs
    if hasattr(self,'groups'):
        self.num_gfpq_param=self.groups
    self.register_buffer('gfpq_param',torch.zeros((self.num_gfpq_param,16),dtype=torch.uint8))
    self.register_buffer('bit_width',torch.zeros(1,dtype=torch.int))
    self.bit_width.fill_(8)

    def hook_bin(layer, inputs):
        with torch.no_grad():
            if hasattr(layer,'weight'):
                layer.weight.org=layer.weight.data.clone()
                layer.weight.data=layer.weight.org
                if hasattr(layer,'groups') and layer.groups>1:
                    layer.bit_width.fill_(16)
                    #list(map(lambda i:hi_gfpq_fake_quant(layer.weight.data[i,:,:,:],int(layer.bit_width.item())),range(self.groups)))
                else:
                    pass
                    #hi_gfpq_fake_quant(layer.weight.data,layer.bit_width.item())

            if hasattr(layer,'groups') and layer.groups>1:
                output_transposed=inputs[0].data.transpose(0,1).contiguous()
                if layer.training:

                    if not layer.initized:
                        if(layer.gfpq_param==0).all():
                            list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_INIT),range(layer.groups)))
                        else:
                            list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(layer.groups)))
                    else:
                        list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(layer.groups)))
                else:
                    list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_APPLY_ONLY),range(layer.groups)))
                inputs[0].data.copy_(output_transposed.data.transpose(0,1))


            else:
                if layer.training:
                    if not layer.initized:
                        if(layer.gfpq_param==0).all():
                            list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_INIT),range(len(inputs))))
                        else:
                            list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(len(inputs))))
                    else:
                        list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(len(inputs))))
                else:
                    list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_APPLY_ONLY),range(len(inputs))))


    self._forward_pre_hooks['hook_bin']=hook_bin

    def hook_restore(layer, inputdata, output):
        if hasattr(layer,'weight'):
            layer.weight.data.copy_(layer.weight.org)
    self._backward_hooks['hook_restore'] = hook_restore

  param_module_class.__init__=__init__
  orig_repr=param_module_class.__repr__
  def __repr__(self):
    return orig_repr(self)+"(bit_width={})".format(self.bit_width.item())
  param_module_class.__repr__=__repr__
  param_module_class.__str__=__repr__

  return param_module_class



def HI_QUANT3(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)
    self.initized=False
    self.num_gfpq_param=1
    if hasattr(self,'n_inputs'):
        self.num_gfpq_param=self.n_inputs
    if hasattr(self,'groups'):
        self.num_gfpq_param=self.groups
    self.register_buffer('gfpq_param',torch.zeros((self.num_gfpq_param,16),dtype=torch.uint8))
    self.register_buffer('bit_width',torch.zeros(1,dtype=torch.int))
    self.bit_width.fill_(8)

    # if hasattr(self,'weight'):
    #     def backward_hook(grad):
    #         grad_input = grad.clone()
    #         self.weight.data[self.weight.org!=0].div_(self.weight.org[self.weight.org!=0])
    #         grad_input[self.weight.org!=0].mul_(self.weight.data[self.weight.org!=0])
    #         return grad_input
    #     self.weight.register_hook(backward_hook)

    def hook_bin(layer, inputs):
        with torch.no_grad():
            if hasattr(layer,'weight'):
                layer.weight.org=layer.weight.data.clone()
                # if hasattr(layer,'groups') and layer.groups>1:
                #     list(map(lambda i:hi_gfpq_fake_quant(layer.weight.data[i,:,:,:],int(layer.bit_width.item())),range(self.groups)))
                # else:
                #     hi_gfpq_fake_quant(layer.weight.data,int(layer.bit_width.item()))
                if layer.bit_width==8:
                    layer.weight.data[layer.weight.org!=0]=2**(layer.weight.data.abs_()[layer.weight.org!=0].log2_().__ilshift__(4).round_().__irshift__(4))
                elif layer.bit_width==16:
                    layer.weight.data[layer.weight.org!=0]=2**(layer.weight.data.abs_()[layer.weight.org!=0].log2_().__ilshift__(7).round_().__irshift__(7))
                layer.weight.data=layer.weight.org.sign()*layer.weight.data

            # if hasattr(layer,'groups') and layer.groups>1:
            #     output_transposed=inputs[0].data.transpose(0,1).contiguous()
            #     if layer.training:
            #         if not layer.initized:
            #             if(layer.gfpq_param==0).all():
            #                 list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_INIT),range(layer.groups)))
            #             else:
            #                 list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(layer.groups)))
            #         else:
            #             list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(layer.groups)))
            #     else:
            #         list(map(lambda i:hi_gfpq_quant_dequant(output_transposed.data[i],layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_APPLY_ONLY),range(layer.groups)))
            #     return (output_transposed.data.transpose(0,1),)

            # else:
            #     if layer.training:
            #         if not layer.initized:
            #             if(layer.gfpq_param==0).all():
            #                 return list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.clone().data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_INIT),range(len(inputs))))
            #             else:
            #                 return list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.clone().data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(len(inputs))))
            #         else:
            #             return list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.clone().data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_UPDATE),range(len(inputs))))
            #     else:
            #         return list(map(lambda i:hi_gfpq_quant_dequant(inputs[i].data.clone().data,layer.gfpq_param[i],bit_width=int(layer.bit_width.item()),mode=GFPQ_MODE_APPLY_ONLY),range(len(inputs))))

    self._forward_pre_hooks['hook_bin']=hook_bin

    def hook_restore(layer, inputdata, output):
        if hasattr(layer,'weight'):
            layer.weight.data.copy_(layer.weight.org)
    self._backward_hooks['hook_restore'] = hook_restore

  param_module_class.__init__=__init__
  orig_repr=param_module_class.__repr__
  def __repr__(self):
    return orig_repr(self)+"(bit_width={})".format(self.bit_width.item())
  param_module_class.__repr__=__repr__
  param_module_class.__str__=__repr__

  return param_module_class


@HI_QUANT
class HiQuantConvTranspose2d(nn.ConvTranspose2d):
    pass
@HI_QUANT
class HiQuantConv2d(nn.Conv2d):
    pass

@HI_QUANT
class HiQuantLinear(nn.Linear):
  pass

@HI_QUANT
class HiQuantConcat(custom_nn.Concat):
  pass

@HI_QUANT
class HiQuantMaxPool2d(nn.MaxPool2d):
  pass


def torch_hi_quant_layer(layer,bit_width=8):
    conv_layer=layer
    if isinstance(layer,torch.nn.Conv2d):
        binary_conv_layer=HiQuantConv2d(
                                        in_channels=conv_layer.in_channels,
                                         out_channels=conv_layer.out_channels,
                                         kernel_size=conv_layer.kernel_size,
                                         stride=conv_layer.stride,
                                         padding=conv_layer.padding,
                                         dilation=conv_layer.dilation,
                                         groups=conv_layer.groups
                                         )

        binary_conv_layer.weight.data=conv_layer.weight.data
        binary_conv_layer.weight.requires_grad=conv_layer.weight.data.requires_grad
        binary_conv_layer.bias=conv_layer.bias
        binary_conv_layer.bit_width.fill_(bit_width)
        return binary_conv_layer

    if isinstance(layer,torch.nn.Linear):
        linear_layer=layer
        binary_linear_layer=HiQuantLinear(
                                        in_features=linear_layer.in_features,
                                         out_features=linear_layer.out_features,
                                         )
        binary_linear_layer.weight.data=linear_layer.weight.data
        binary_linear_layer.bias.data=linear_layer.bias.data
        binary_linear_layer.bit_width.fill_(bit_width)
        return binary_linear_layer

    if isinstance(layer,torch.nn.ConvTranspose2d):
        deconv_layer=layer
        binary_deconv_layer=HiQuantConvTranspose2d(
                                        in_channels=deconv_layer.in_channels,
                                         out_channels=deconv_layer.out_channels,
                                         kernel_size=deconv_layer.kernel_size,
                                         stride=deconv_layer.stride,
                                         padding=deconv_layer.padding,
                                         dilation=deconv_layer.dilation,
                                         groups=deconv_layer.groups
                                         )
        binary_deconv_layer.weight.data=deconv_layer.weight.data
        binary_deconv_layer.bias.data=deconv_layer.bias.data
        binary_deconv_layer.bit_width.fill_(bit_width)
        return binary_deconv_layer


    if isinstance(layer,custom_nn.Concat):
        concat_layer=layer
        binary_concat_layer=HiQuantConcat(axis=concat_layer.axis,
                                          n_inputs=concat_layer.n_inputs
                                         )
        binary_concat_layer.bit_width.fill_(bit_width)
        return binary_concat_layer

    if isinstance(layer,nn.MaxPool2d):
        pool_layer=layer
        binary_pool_layer=HiQuantMaxPool2d(kernel_size=pool_layer.kernel_size,
                                          stride=pool_layer.stride,
                                          padding=pool_layer.padding,
                                          dilation=pool_layer.dilation,
                                          ceil_mode=pool_layer.ceil_mode,
                                         )
        binary_pool_layer.bit_width.fill_(bit_width)
        return binary_pool_layer

def torch_hi_quant_layers(layers,keys=[],pattern=None,include=[],exclude=[],bit_width=8):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()

    for i,key in enumerate(keys):
        if key in exclude:
            continue

        if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d) or isinstance(layers._modules[key],torch.nn.MaxPool2d) or isinstance(layers._modules[key],torch.nn.ConvTranspose2d) or isinstance(layers._modules[key],custom_nn.Concat):
            if not(pattern is None) and pattern in key:
                continue
            layer=layers._modules[key]
            layers._modules[key]=torch_hi_quant_layer(layer,bit_width=bit_width)
            del layer

        else:
            seq=layers._modules[key]
            torch_hi_quant_layers(seq,pattern=pattern,bit_width=bit_width)


def lowrank_decomposition_conv_layer(layer,key,percentale,more_k=0,has_bn=True):
        W=layer.weight.data.numpy()
        b=layer.bias.data.numpy() if layer.bias is not None else 0
        oc,ic,kernel_h,kernel_w=W.shape
        pad_h,pad_w=layer.padding
        num_groups = 1
        N, C, D, D = W.shape

       
#         print N,C,D,D
        #####################
        #print N,C,D,K
        # SVD approximation
        for g in xrange(num_groups):
            W_ = W[N*g:N*(g+1)].transpose(1, 2, 3, 0).reshape((C*D, D*N))
#             print W_.shape
            U, S, V = np.linalg.svd(W_)
            
            temp1 = np.cumsum(S) / sum(S)
            idx = np.where(temp1 >= percentale)
           
            ####
            K=idx[0][0]
            # K=C
            #/8+1)*8 if (idx[0][0]/8+1)*8>(N*0.5) else N*4


            if C==3:
                K=8
            elif C==1:
                K=3
            
            # if S.shape[0]>K:
            #     CK=np.zeros((K,S.shape[1]))
            #     CK[:K]=S[:K]
            # else:
            #     CK=np.zeros((S.shape[0],S.shape[1]))
            #     CK[:K]=S[:K]
            print(U.shape,S.shape,V.shape,K)
            try:
                v = U[:, :K] * np.sqrt(S[:K])
            except Exception as e:
                temp=np.zeros([K])
                temp[:S.shape[0]]=S[:]
                v = U[:, :K] * np.sqrt(temp)
                S=temp
                

            v_weights=np.zeros((K,C,D,1))
            v_bias=np.zeros((K))
            h_weights=np.zeros((N,K,1,D))
            h_bias=np.zeros((N))
            v_bias[...] = 0
            h_bias[...] = b
            ####
            # print'map',(idx[0][0])
            # print(sum(S[:K]) / sum(S))
            # print v.shape
            # print C*D*K
            #print(v.shape,C,D,K)
            v = v[:, :K].reshape((C, D, 1, K)).transpose(3, 0, 1, 2)
            v_weights[K*g:K*(g+1)] = v.copy()
            print(V.shape,K,S.shape)
            try:
                h = V[:K, :] * np.sqrt(S)[:K, np.newaxis]
            except Exception as e:
                temp=np.zeros([K,K])
                temp[:V.shape[0],:V.shape[1]]=V[:,:]
                h = temp * np.sqrt(S)[:K, np.newaxis]
            print(h.shape,K,D,N)
            try:
                h = h.reshape((K, 1, D, N)).transpose(3, 0, 1, 2)
            except Exception as e:
                h = h[:,:D*N].reshape((K, 1, D, N)).transpose(3, 0, 1, 2)

            h_weights[N*g:N*(g+1)] = h.copy()
#             print layer.padding,'pad',layer.stride,layer.kernel_size
#             print ic,K
        v_conv_layer=torch.nn.Conv2d(in_channels=ic,out_channels=K,kernel_size=(layer.kernel_size[0],1),padding=(layer.padding[0],0),stride=(layer.stride[0], 1))
        #torch.nn.init.xavier_uniform(v_conv_layer.weight.data)
        v_conv_layer.weight.data[:,:,:,:]=torch.from_numpy(v_weights).float()
        v_conv_layer.bias.data=torch.zeros_like(v_conv_layer.bias.data)

        h_conv_layer=torch.nn.Conv2d(in_channels=K,out_channels=oc,kernel_size=(1,layer.kernel_size[1]),padding=(0,layer.padding[1]),stride=(1, layer.stride[1]))
        #torch.nn.init.xavier_uniform(h_conv_layer.weight.data)
        h_conv_layer.weight.data[:,:,:,:]=torch.from_numpy(h_weights).float()

        h_conv_layer.bias.data=torch.from_numpy(h_bias).float()
        layer_list=[]
        layer_list.append((key+'_v',v_conv_layer))
        if(has_bn):
            layer_list.append((key+'_v_bn',torch.nn.BatchNorm2d(K)))
        layer_list.append((key+'_h',h_conv_layer))

        new_layers = OrderedDict(layer_list)
        return nn.Sequential(new_layers)
xrange=range
def torch_lowrank_layers(layers,keys=[],percentale=0.9,more_k=True,has_bn=True):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    #print(layers)
    for i,key in enumerate(keys):
        if isinstance(layers._modules[key],torch.nn.Conv2d) or isinstance(layers._modules[key],BinaryConv2d)or isinstance(layers._modules[key],IRConv2d):

            conv_layer=layers._modules[key]
            # print conv_layer
            if conv_layer.kernel_size[0]==1 or conv_layer.kernel_size[1]==1:
                continue
#             print conv_layer
            decomposed=lowrank_decomposition_conv_layer(conv_layer,key,more_k=more_k,percentale=percentale,has_bn=has_bn)
            del conv_layer
            layers._modules[key]=decomposed
        else:#if isinstance(layers._modules[key],torch.nn.Sequential):
            seq=layers._modules[key]
            torch_lowrank_layers(seq,percentale=percentale,more_k=more_k,has_bn=has_bn)

class LAMBDA(nn.Module):
    
    def __init__(self):
        super(LAMBDA,self).__init__()
        pass
    def forward(self,x):
        return x
def torch_mergebn_layers(layers,removed=True,keys=[]):
    if keys==[]:
        if layers is None:
            return
        keys=list(layers._modules.keys())
    last_conv_key=None
    for i,key in enumerate(keys):

        if isinstance(layers._modules[key],torch.nn.Conv2d)or isinstance(layers._modules[key],torch.nn.Sequential):
            last_conv_key=key
        if isinstance(layers._modules[key],torch.nn.BatchNorm2d):

            bn_layer=layers._modules[key]
            bn_weight=bn_layer.weight.data
            bn_bias=bn_layer.bias.data
            running_var=bn_layer.running_var.data
            running_mean=bn_layer.running_mean.data
            eps=bn_layer.eps

            last_conv_layer=layers._modules[last_conv_key][-1]if isinstance(layers._modules[last_conv_key],torch.nn.Sequential) else layers._modules[last_conv_key]
            tmp=bn_weight/torch.sqrt(running_var+eps)
            last_conv_layer.weight.data=tmp.view(-1,1,1,1)*last_conv_layer.weight.data
            try:
                last_conv_layer.bias.data=tmp*(last_conv_layer.bias.data-running_mean)+bn_bias
            except Exception as e:
                last_conv_layer.bias=torch.nn.Parameter(tmp*(torch.zeros_like(running_mean)-running_mean)+bn_bias)

            if removed:
                del bn_layer
                layers._modules.pop(key)
        else:
            seq=layers._modules[key]
            torch_mergebn_layers(seq,removed=removed)
# def torch_mergebn_layers(layers,keys=[]):
#     if keys==[]:
#         if layers is None:
#             return
#         keys=list(layers._modules.keys())
#     last_conv_key=None
#     for i,key in enumerate(keys):

#         if isinstance(layers._modules[key],torch.nn.Conv2d)or isinstance(layers._modules[key],BinaryConv2d) or isinstance(layers._modules[key],torch.nn.Sequential)or isinstance(layers._modules[key],IRConv2d):
#             last_conv_key=key
#         if isinstance(layers._modules[key],torch.nn.BatchNorm2d):
#             bn_layer=layers._modules[key]
#             bn_weight=bn_layer.weight.data
#             bn_bias=bn_layer.bias.data
#             running_var=bn_layer.running_var.data
#             running_mean=bn_layer.running_mean.data
#             eps=bn_layer.eps

#             last_conv_layer=layers._modules[last_conv_key][-1]if isinstance(layers._modules[last_conv_key],torch.nn.Sequential) else layers._modules[last_conv_key]
#             tmp=bn_weight/torch.sqrt(running_var+eps)
#             last_conv_layer.weight.data=tmp.view(-1,1,1,1)*last_conv_layer.weight.data
#             bias_data=last_conv_layer.bias if last_conv_layer.bias is not None else 0
#             last_conv_layer.bias.data=tmp*(bias_data-running_mean)+bn_bias
#             #del bn_layer
#             layers._modules[key]=LAMBDA()
#         else:
#             seq=layers._modules[key]
#             torch_mergebn_layers(seq)

def sign_binary(data):
    return math.sqrt(2. / (data.shape[1] * data.shape[2] * data.shape[3])) * data.sign()
def torch_mergebn_layers_sign_binary(layers,keys=[]):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    last_conv_key=None
    for i,key in enumerate(keys):

        if isinstance(layers._modules[key],torch.nn.Conv2d)or isinstance(layers._modules[key],torch.nn.Sequential):
            last_conv_key=key
        if isinstance(layers._modules[key],torch.nn.BatchNorm2d):
            bn_layer=layers._modules[key]
            bn_weight=bn_layer.weight.data
            bn_bias=bn_layer.bias.data
            running_var=bn_layer.running_var.data
            running_mean=bn_layer.running_mean.data
            eps=bn_layer.eps

            last_conv_layer=layers._modules[last_conv_key][-1]if isinstance(layers._modules[last_conv_key],torch.nn.Sequential) else layers._modules[last_conv_key]
            tmp=bn_weight/torch.sqrt(running_var+eps)
            print(sign_binary(last_conv_layer.weight.data))
            last_conv_layer.weight.data=tmp.view(-1,1,1,1)*sign_binary(last_conv_layer.weight.data)
            last_conv_layer.bias.data=tmp*(last_conv_layer.bias.data-running_mean)+bn_bias
            del bn_layer
            layers._modules.pop(key)
        else:
            seq=layers._modules[key]
            torch_mergebn_layers_sign_binary(seq)


def torch_replace_bn_with_scale_layers(layers,keys=[]):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    last_conv_key=None
    for i,key in enumerate(keys):

        if isinstance(layers._modules[key],torch.nn.Conv2d)or isinstance(layers._modules[key],torch.nn.Sequential):
            last_conv_key=key
        if isinstance(layers._modules[key],torch.nn.BatchNorm2d):
            bn_layer=layers._modules[key]
            bn_weight=bn_layer.weight.data
            bn_bias=bn_layer.bias.data
            running_var=bn_layer.running_var.data
            running_mean=bn_layer.running_mean.data
            eps=bn_layer.eps

            last_conv_layer=layers._modules[last_conv_key][-1]if isinstance(layers._modules[last_conv_key],torch.nn.Sequential) else layers._modules[last_conv_key]
            tmp=bn_weight/torch.sqrt(running_var+eps)
            print(sign_binary(last_conv_layer.weight.data))
            last_conv_layer.weight.data=tmp.view(-1,1,1,1)*sign_binary(last_conv_layer.weight.data)
            last_conv_layer.bias.data=tmp*(last_conv_layer.bias.data-running_mean)+bn_bias

            scale_layer=nn.BatchNorm2d(bn_weight.data.shape[0], affine=False)
            scale_layer.weight.data=tmp
            del bn_layer

            layers._modules[key]=nn.Scale()
        else:
            seq=layers._modules[key]
            torch_replace_bn_with_scale_layers(seq)
def torch_binary_layer(layer,optimizer=None):
    conv_layer=layer
    if isinstance(layer,torch.nn.Conv2d):
        if optimizer is  None:
            # BinaryConv2d
            # IRConv2d
            binary_conv_layer=BinaryConv2d(
                                            in_channels=conv_layer.in_channels,
                                             out_channels=conv_layer.out_channels,
                                             kernel_size=conv_layer.kernel_size,
                                             stride=conv_layer.stride,
                                             padding=conv_layer.padding,
                                             dilation=conv_layer.dilation,
                                             groups=conv_layer.groups
                                             )
        else:
            binary_conv_layer=BinaryConv2d(
                                            in_channels=conv_layer.in_channels,
                                             out_channels=conv_layer.out_channels,
                                             kernel_size=conv_layer.kernel_size,
                                             stride=conv_layer.stride,
                                             padding=conv_layer.padding,
                                             dilation=conv_layer.dilation,
                                             groups=conv_layer.groups,
                                             optimizer=optimizer
                                             )
        binary_conv_layer.weight.data=conv_layer.weight.data
        binary_conv_layer.bias=conv_layer.bias
        return binary_conv_layer

    if isinstance(layer,torch.nn.Linear):
        linear_layer=layer
        binary_linear_layer=BinaryLinear(
                                        in_features=linear_layer.in_features,
                                         out_features=linear_layer.out_features,
                                         )
        binary_linear_layer.weight.data=linear_layer.weight.data
        binary_linear_layer.bias.data=linear_layer.bias.data
        return binary_linear_layer


def torch_binarize_layers(layers,keys=[],exclude=[],pattern=None):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()

    for i,key in enumerate(keys):
        if key in exclude:

            continue
        if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d):
            if not(pattern is None) and pattern in key:
                continue
            layer=layers._modules[key]
            layers._modules[key]=torch_binary_layer(layers._modules[key])
            del layer

        else:
            seq=layers._modules[key]
            torch_binarize_layers(seq,exclude=exclude)

def torch_binarize_layers_hook(layers,keys=[],pattern=None):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    for i,key in enumerate(keys):

        if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d):
            if not(pattern is None) and pattern in key:
                continue

            mean_axes=(1,2,3)
            reshape_size=(-1,1,1,1)

            def hook_bin(layer, inputdata):

                    if not hasattr(layer.weight,'org'):
                        layer.weight.org=layer.weight.data.clone()
                    layer.weight.data=layer.weight.org.sign()#math.sqrt(2. / (layer.weight.org.shape[1] * layer.weight.org.shape[2] * layer.weight.org.shape[3])) * layer.weight.org.sign()

            def hook_restore(layer, inputdata, output):
                if hasattr(layer.weight,'grad')and layer.weight.grad is not None:
                    layer.weight.grad.clamp_(-1,1)#[1<torch.abs(layer.weight.org)]=0
                layer.weight.data.copy_(layer.weight.org)

            layers._modules[key].register_forward_pre_hook(hook_bin)
            layers._modules[key].register_backward_hook(hook_restore)


        else:
            seq=layers._modules[key]
            torch_binarize_layers_hook(seq)






def freeze(layers,keys=[],freeze_layer=[]):
    if keys==[]:
        if layers is None:
            return
        keys=layers.module._modules.keys()

    for i,key in enumerate(keys):
        if key not in freeze_layer:
            continue
        for name,param in layers.module._modules[key].named_parameters():
            param.requires_grad=False



def unfreeze(model):
    for name,param in model.named_parameters():
        param.requires_grad=True

def show_requires_grad(model):
    for name,param in model.named_parameters():
        print(name,param.requires_grad)

def show_param_mean(model):
    for name,param in model.named_parameters():
        print(name,param.mean())
# def torch_clip_weight(layer):
#     if isinstance(layer,torch.nn.Conv2d):
#         conv_layer=layer
#         conv_layer.weight.data=conv_layer.weight.data
#         return binary_conv_layer

#     if isinstance(layer,torch.nn.Linear):
#         linear_layer=layer
#         binary_linear_layer.weight.data=linear_layer.weight.data
#         binary_linear_layer.bias.data=linear_layer.bias.data
#         return binary_linear_layer


# def torch_clip_weights(layers,keys=[]):
#     if keys==[]:
#         keys=layers._modules.keys()

#     for i,key in enumerate(keys):
#         if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d):
#             torch_clip_weight(layers._modules[key])

#         else:#if isinstance(layers._modules[key],torch.nn.Sequential):
#             seq=layers._modules[key]
#             torch_binarize_layers(seq)



class ActQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,scale,num_bits=8):
        ctx.minval=-(1<<(num_bits-1));
        ctx.maxval=(1<<(num_bits-1))-1;
        ctx.save_for_backward(x,scale)
        output=torch.clamp(torch.round(x/scale),ctx.minval,ctx.maxval)*scale


        return output
    @staticmethod
    def backward(ctx, g):
        # print(len(g))
        # exit(0)
        g_x, g_s = None, None
        x,scale = ctx.saved_variables
        x.div_(scale)
        round_val=torch.round(x)

        mask=(round_val>=ctx.minval)&(round_val<=ctx.maxval)
        if ctx.needs_input_grad[0]:
            g_x=torch.where(mask,g,torch.Tensor([0]))
        if ctx.needs_input_grad[1]:
            round_val.clamp_(ctx.minval,ctx.maxval)
            s_l=torch.where(mask,round_val-x,round_val)
            g_s=torch.dot(s_l.view(-1),g.view(-1)).view(-1)

        return g_x,g_s,None

class ParamQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,num_bits=8,is_winograd=False):
        dim=len(w.shape)
        quant_min_value=-(1<<(num_bits-1));
        quant_max_value=(1<<(num_bits-1))-1;
        n=w.shape[0]
        if is_winograd and num_bits==8:
            threshold=31
        else:
            threshold=(1<<(num_bits-1)-1);

        temp,_=torch.max(torch.abs(w).view(n,-1),dim=1)

        temp=temp.view([-1]+[1]*(dim-1))
        scale=threshold/temp

        output=torch.clamp(torch.round(w*scale),quant_min_value,quant_max_value)/scale
        return output

    @staticmethod
    def backward(ctx, g):

        return g,None,None

act_quant=ActQuant.apply
param_quant=ParamQuant.apply


from torch.nn.parameter import Parameter




def Quant(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)
    assert hasattr(self,'weight')
    def hook_bin(layer, inputdata):
        with torch.no_grad():
            # if not hasattr(layer.weight,'org'):
            layer.weight.org=layer.weight.data.clone()
            layer.weight.data==ParamQuant.apply(layer.weight.org,8,False)
    self._forward_pre_hooks['hook_bin']=hook_bin
    def hook_restore(layer, inputdata, output):
        layer.weight.data.copy_(layer.weight.org)
    self._backward_hooks['hook_restore'] = hook_restore
  param_module_class.__init__=__init__
  return param_module_class

@Quant
class QuantConv2d(nn.Conv2d):
    pass

class ActQuantLayer(nn.Module):
    def __init__(self,num_bits):
        super(ActQuantLayer, self).__init__()
        self.register_parameter('output_scale',Parameter(torch.Tensor(1)))
        self.num_bits=num_bits

    def forward(self, x):

        if not hasattr(self,'quant_output'):
            self.register_buffer('quant_output',x.clone())
        else:
            self.quant_output=x.clone()

        calib=False
        if calib:
            temp,_=torch.max(torch.abs(x).view(n,-1),dim=1)
            self.output_scale.data=temp.mean()/127.0*(1-0.9)+self.output_scale.data*0.9
            output=x
        else:
            output=ActQuant.apply(self.quant_output,self.output_scale,self.num_bits)
        return output


def torch_quant_layers_hook(layers,num_bits=8,momentum=0.99,is_cali_mode=True,keys=[],pattern=None):

    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    for i,key in enumerate(keys):
        if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d):
            if not(pattern is None) and pattern in key:
                continue

            def hook_quant_weight(layer, inputdata):
                layer.weight.ori=layer.weight.clone()
                layer.weight.data=ParamQuant.apply(layer.weight.ori,num_bits,False)

            def hook_quant_restore_weight(layer, inputdata, output):
                layer.weight.data.copy_(layer.weight.ori)
            layers._modules[key].register_forward_pre_hook(hook_quant_weight)
            layers._modules[key].register_backward_hook(hook_quant_restore_weight)

        if isinstance(layers._modules[key],torch.nn.BatchNorm2d):
            def hook_quant_act(layer, input,output):

                if not hasattr(layer,'output_scale'):
                    layer.register_parameter('output_scale',Parameter(torch.zeros(1)))
                if  is_cali_mode:
                    with torch.no_grad():
                        n=output[0].shape[0]
                        b_s=output[0].abs().view(n,-1).max(1)[0].mean()/127.0
                        layer.output_scale.data=layer.output_scale.data*momentum+(1-momentum)*b_s
                        print(layer.output_scale.data)
                else:
                    if not hasattr(layer,'quant_output'):
                        layer.register_buffer('quant_output',output[0].clone())
                    else:
                        layer.quant_output=output[0].clone()
                    output[0]=ActQuant.apply(layer.quant_output,layer.output_scale,num_bits)
            layers._modules[key].register_forward_hook(hook_quant_act)
        else:
            seq=layers._modules[key]
            torch_quant_layers_hook(seq)


# import torchvision.models as models
# vgg16 = models.resnet50(pretrained=False)
# torch_hi_quant_layers(vgg16)

# pred=vgg16.forward(torch.rand(1,3,224,224))
# pred.backward(torch.ones_like(pred))
# pred=vgg16.forward(torch.rand(1,3,224,224))
# pred.backward(torch.ones_like(pred))
# print(vgg16)