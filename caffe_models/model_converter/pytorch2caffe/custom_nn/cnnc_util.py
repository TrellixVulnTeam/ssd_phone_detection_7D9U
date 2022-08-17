import torch.nn as nn
import torch
import torch.nn.functional as F
from  collections import OrderedDict
import numpy as np
import math

class ForwardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.mean(torch.abs(w),(1,2,3)).view(-1,1,1,1)*w.sign()
        # torch.reshape(mean,reshape_size)
        # return math.sqrt(2. / (w.shape[1] * w.shape[2] * w.shape[3])) * w.sign()

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
    mean_axes=tuple(range(1,dim))
    reshape_size=[-1]+[1]*(dim-1)
    def hook_bin(layer, inputdata):

        #layer.weight.data=torch.clamp(layer.weight.data,-1,1)
        with torch.no_grad():
            if not hasattr(layer.weight,'org'):

                layer.weight.org=layer.weight.data.clone()
            #layer.weight.org=layer.weight.data.clone()
            sign=torch.sign(layer.weight.org)
            #layer.weight.data=sign
            #print("binarization")
            layer.weight.data=torch.reshape(torch.mean(torch.abs(layer.weight.org),mean_axes),reshape_size)*sign



    self._forward_pre_hooks['hook_bin']=hook_bin
    def hook_restore(layer, inputdata, output):
        # print"restore"
        # if layer.clip_grad:
            # if hasattr(layer.weight,'grad')and layer.weight.grad is not None:
                # layer.weight.grad[torch.abs(layer.weight.org)>torch.abs(layer.weight.data)]=0

        layer.weight.data.copy_(layer.weight.org)
    self._backward_hooks['hook_restore'] = hook_restore
  param_module_class.__init__=__init__
  return param_module_class

# add binary weight network decorator
def Tri(param_module_class):
  orig_init=param_module_class.__init__
  def __init__(self,*args,**kws):
    orig_init(self,*args,**kws)
    assert hasattr(self,'weight')
    def hook_bin(layer, inputdata):
        with torch.no_grad():
            if not hasattr(layer.weight,'org'):
                layer.weight.org=layer.weight.data.clone()
            layer.weight.data=torch.triu(layer.weight.org)
            #print(layer.weight.data.shape)
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
@BWN
class BinaryConv2d(nn.Conv2d):
    pass

@BWN
class BinaryLinear(nn.Linear):
  pass

@Tri
class TriLinear(nn.Linear):
  pass





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
#             print idx[0][0]
            ####
            K=idx[0][0]#/8+1)*8 #if (idx[0][0]/8+1)*8>(N*0.5) else N*4
            if C==3:
                K=8
            elif C==1:
                K=3
            v = U[:, :K] * np.sqrt(S[:K])
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
            v = v[:, :K].reshape((C, D, 1, K)).transpose(3, 0, 1, 2)
            v_weights[K*g:K*(g+1)] = v.copy()
            h = V[:K, :] * np.sqrt(S)[:K, np.newaxis]
            h = h.reshape((K, 1, D, N)).transpose(3, 0, 1, 2)
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

def torch_lowrank_layers(layers,keys=[],percentale=0.9,more_k=True,has_bn=True):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()
    #print(layers)
    for i,key in enumerate(keys):
        if isinstance(layers._modules[key],torch.nn.Conv2d):

            conv_layer=layers._modules[key]
            # print conv_layer
            if conv_layer.kernel_size[0]==1 or conv_layer.kernel_size[0]==1:
                continue
#             print conv_layer
            decomposed=lowrank_decomposition_conv_layer(conv_layer,key,more_k=more_k,percentale=percentale,has_bn=has_bn)
            del conv_layer
            layers._modules[key]=decomposed
        else:#if isinstance(layers._modules[key],torch.nn.Sequential):
            seq=layers._modules[key]
            torch_lowrank_layers(seq,percentale=percentale,more_k=more_k,has_bn=has_bn)


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
            last_conv_layer.bias.data=tmp*(last_conv_layer.bias.data-running_mean)+bn_bias

            if removed:
                del bn_layer
                layers._modules.pop(key)
        else:
            seq=layers._modules[key]
            torch_mergebn_layers(seq,removed=removed)

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


def torch_binarize_layers(layers,keys=[],optimizer=None,pattern=None):
    if keys==[]:
        if layers is None:
            return
        keys=layers._modules.keys()

    for i,key in enumerate(keys):
        if isinstance(layers._modules[key],torch.nn.Linear) or isinstance(layers._modules[key],torch.nn.Conv2d):
            if not(pattern is None) and pattern in key:
                continue

            layer=layers._modules[key]
            layers._modules[key]=torch_binary_layer(layers._modules[key],optimizer=optimizer)
            del layer
#             print key
#             if step:
#                 return

        else:#if isinstance(layers._modules[key],torch.nn.Sequential):
            seq=layers._modules[key]
            torch_binarize_layers(seq,optimizer=optimizer)

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
            # if '_' in key:
            def hook_bin(layer, inputdata):
                #with torch.no_grad():
                    # L=optimizer.state#+ 1e-8 if optimizer.state[layer.weight].has_key('exp_avg_sq') else 1e-8
                    # print id(layer.weight) in map(id,L)
                    #print L.has_key('exp_avg_sq')
                    if not hasattr(layer.weight,'org'):
                        layer.weight.org=layer.weight.data.clone()
                    layer.weight.data=layer.weight.org.sign()#math.sqrt(2. / (layer.weight.org.shape[1] * layer.weight.org.shape[2] * layer.weight.org.shape[3])) * layer.weight.org.sign()
                    # print(layer.weight.data)
                    #layer.weight.data=torch.mean(torch.abs(layer.weight.org),(1,2,3)).view(-1,1,1,1)*layer.weight.org.sign()
                    # print(layer.weight.data)
                    # sign=torch.sign(layer.weight.org)
                    # print torch.sum(torch.abs(layer.weight.org*L),dim=mean_axes,keepdim=True).shape
                    # layer.weight.data=torch.sum(torch.abs(layer.weight.org*L),dim=mean_axes,keepdim=True)/torch.sum(L,dim=mean_axes,keepdim=True)*sign
                    # layer.weight=torch.nn.Parameter(ForwardSign.apply(layer.weight))

            def hook_restore(layer, inputdata, output):
                if hasattr(layer.weight,'grad')and layer.weight.grad is not None:
                    layer.weight.grad.clamp_(-1,1)#[1<torch.abs(layer.weight.org)]=0
                layer.weight.data.copy_(layer.weight.org)

            layers._modules[key].register_forward_pre_hook(hook_bin)
            layers._modules[key].register_backward_hook(hook_restore)

            # if '_v' in key:
                # def hook_bin(layer, inputdata):
                        # if not hasattr(layer.weight,'org'):
                            # layer.weight.org=layer.weight.data.clone()
                        # layer.weight.data=layer.weight.org.clamp(-1,1).sign()#math.sqrt(2. / (layer.weight.org.shape[1] * layer.weight.org.shape[2] * layer.weight.org.shape[3])) *
                # def hook_restore(layer, inputdata, output):
                    # if hasattr(layer.weight,'grad')and layer.weight.grad is not None:
                        # layer.weight.grad[1<torch.abs(layer.weight.org)]=0
                    # layer.weight.data.copy_(layer.weight.org)

                    # layers._modules[key].register_forward_pre_hook(hook_bin)
                    # layers._modules[key].register_backward_hook(hook_restore)
        else:
            seq=layers._modules[key]
            torch_binarize_layers_hook(seq)

def freeze(model):
    for name,param in model.named_parameters():
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
            if not hasattr(layer.weight,'org'):
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

                if not hasattr(layer.weight,'ori'):
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
# torch_quant_layers_hook(vgg16)
# pred=vgg16.forward(torch.rand(1,3,224,224))
# pred.backward(torch.ones_like(pred))
# pred=vgg16.forward(torch.rand(1,3,224,224))
# pred.backward(torch.ones_like(pred))
# # print(vgg16)