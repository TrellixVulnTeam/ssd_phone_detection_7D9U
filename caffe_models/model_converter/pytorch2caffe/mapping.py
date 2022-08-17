
def kwargs_process(*key_value):
    key,value=key_value
    if key=="shape":
        return "  shape{\n%s\n  }"%("\n".join(["  dim:%s"%(dim,) for dim in value]))
    elif "filler" in key:
        return '  %s{\n  type:"%s"\n  }'%(key, value)
    else:
        return "  %s:%s"%(key,value)

def create_layer(layer_type,layer_name,bottom_names,top_name,**kwags):
    params_str=' %s_param:{\n%s\n }\n'%(layer_type.lower(),"\n".join(map(lambda x:kwargs_process(x,kwags[x]),kwags))) if len(kwags)>0 else ""
    bottoms_str="\n".join(map(lambda x:' bottom:"%s"'%x,bottom_names))
    top_str=' top:"%s"'%top_name
    layer_str='layer{\n name:"%s"\n type:"%s"\n%s\n%s\n%s}\n'%(layer_name,layer_type,bottoms_str,top_str,params_str)
    return layer_str

def create_layer_with_param(layer_type,layer_name,bottom_names,top_name,lr_mults,**kwags):
    params_str=' %s_param:{\n%s\n }\n'%(layer_type.lower(),"\n".join(map(lambda x:kwargs_process(x,kwags[x]),kwags))) if len(kwags)>0 else ""
    bottoms_str="\n".join(map(lambda x:' bottom:"%s"'%x,bottom_names))
    top_str=' top:"%s"'%top_name
    param_str="\n".join(map(lambda x:' param{lr_mult: %s}'%x,lr_mults))
    layer_str='layer{\n name:"%s"\n type:"%s"\n %s \n  %s\n%s\n%s}\n'%(layer_name,layer_type,param_str,bottoms_str,top_str,params_str)
    return layer_str

def create_input(layer_name,config,bottom_names,top_name,shape):
    return create_layer("Input",layer_name,bottom_names,top_name,shape=shape)


def create_relu(layer_name,config,bottom_names,top_name):
    return create_layer("ReLU",layer_name,bottom_names,top_name)

def create_concat(layer_name,config,bottom_names,top_name):
    return create_layer("Concat",layer_name,bottom_names,top_name,axis=config.get('axis',1))

def create_softmax(layer_name,config,bottom_names,top_name):
    return create_layer("Softmax",layer_name,bottom_names,top_name)

# def create_conv2dx(layer_name,config,bottom_names,top_name):
#     return create_layer("Conv2dX",layer_name,bottom_names,top_name)

# def create_nconv2dx(layer_name,config,bottom_names,top_name):
#     return create_layer("NConv2dX",layer_name,bottom_names,top_name)

# def create_down(layer_name,config,bottom_names,top_name):
#     return create_layer("Down",layer_name,bottom_names,top_name)

# def create_up(layer_name,config,bottom_names,top_name):
#     return create_layer("Up",layer_name,bottom_names,top_name)

# def create_softargmax(layer_name,config,bottom_names,top_name):
#     return create_layer("SoftArgmax",layer_name,bottom_names,top_name)

# def create_l2norm(layer_name,config,bottom_names,top_name):
#     scale=config['use_weight']
#     epsilon=config['eps']
#     l2norm_str=create_layer("Normalize",layer_name,bottom_names,top_name)
#     scale_str='\n'+create_layer('Scale',layer_name+'_s',bottom_names,top_name,bias_term=False) if scale else ""
#     return l2norm_str+scale_str

def create_l2norm(layer_name,config,bottom_names,top_name):
    scale=not config['use_weight']
    eps=config['eps']
    return create_layer("Normalize",layer_name,bottom_names,top_name,eps=eps,across_spatial=False,scale_filler="constant",channel_shared=scale).replace("normalize_param","norm_param")

def create_maxpool2d(layer_name,config,bottom_names,top_name):
    kernel_h,kernel_w=config['kernel_size'],config['kernel_size']
    stride_h,stride_w=config['stride'],config['stride']
    pad_h,pad_w=config['padding'],config['padding']
    pool="MAX"
    return create_layer('Pooling',layer_name,bottom_names,top_name,pool=pool,kernel_h=kernel_h,kernel_w=kernel_w,stride_h=stride_h,stride_w=stride_w,pad_h=pad_h,pad_w=pad_w)

def create_adaptiveavgpool2d(layer_name,config,bottom_names,top_name):
    input_shape=config['bottoms'][0][1]
    ih,iw=input_shape[2],input_shape[3]
    oh,ow=config['output_size']
    kernel_h,kernel_w=ih//oh,iw//ow
    stride_h,stride_w=kernel_h,kernel_w
    pad_h,pad_w=0,0
    pool="AVE"
    return create_layer('Pooling',layer_name,bottom_names,top_name,pool=pool,kernel_h=kernel_h,kernel_w=kernel_w,stride_h=stride_h,stride_w=stride_w,pad_h=pad_h,pad_w=pad_w)

def create_batchnorm2d(layer_name,config,bottom_names,top_name):
    scale=config['affine']
    epsilon=config['eps']
    moving_average_fraction=config['momentum']
    bias_term=config['use_bias']
    bn_str=create_layer('BatchNorm',layer_name,bottom_names,bottom_names[0],eps=epsilon,moving_average_fraction=moving_average_fraction).replace('batchnorm_param','batch_norm_param')
    scale_str='\n'+create_layer('Scale',layer_name+'_s',bottom_names,top_name,bias_term=bias_term) if scale else ""
    return bn_str+scale_str

def create_upsample(layer_name,config,bottom_names,top_name):
    if config['mode']=='bilinear':
        size=config['size']
        return create_layer("Interp",layer_name,bottom_names,top_name,width=size[0],height=size[1])
    else:
        scale=config['size'][0]//config['bottoms'][0][1][3]
        return create_layer('Upsample',layer_name,bottom_names,top_name,scale=scale)


def create_aspp(layer_name,config,bottom_names,top_name):
    return create_layer("ASPP",layer_name,bottom_names,top_name)

def create_deeplabv3head(layer_name,config,bottom_names,top_name):
    return create_layer("deeplabv3head",layer_name,bottom_names,top_name)

def create_conv2d(layer_name,config,bottom_names,top_name):
    kernel_h,kernel_w=config['kernel_size']
    num_output=config['out_channels']
    stride_h,stride_w=config['stride']
    dilation=config['dilation']
    group=config['groups']
    pad_h,pad_w=config['padding']
    bias_term='true' if config['use_bias'] else 'false'
    return create_layer('Convolution',layer_name,bottom_names,top_name,kernel_h=kernel_h,kernel_w=kernel_w,stride_h=stride_h,stride_w=stride_w,pad_h=pad_h,pad_w=pad_w,dilation=dilation[0],bias_term=bias_term,num_output=num_output,group=group)


def create_convtranspose2d(layer_name,config,bottom_names,top_name):
    kernel_h,kernel_w=config['kernel_size']
    group=config['groups']
    num_output=config['out_channels']
    stride_h,stride_w=config['stride']
    dilation=config['dilation']
    pad_h,pad_w=config['padding']
    bias_term='true' if config['use_bias'] else 'false'
    return create_layer('Deconvolution',layer_name,bottom_names,top_name,kernel_h=kernel_h,kernel_w=kernel_w,stride_h=stride_h,stride_w=stride_w,pad_h=pad_h,pad_w=pad_w,dilation=dilation[0],bias_term=bias_term,num_output=num_output,group=group).replace('deconvolution_param','convolution_param')

def create_dropout(layer_name,config,bottom_names,top_name):
    return create_layer("Dropout",layer_name,bottom_names,top_name,dropout_ratio=config["p"])

def create_permute(layer_name,config,bottom_names,top_name):

    return create_layer("Permute",layer_name,bottom_names,top_name,order=config['order'])

def create_flatten(layer_name,config,bottom_names,top_name):
    return create_layer("Flatten",layer_name,bottom_names,top_name)

def create_binaryconv2d(layer_name,config,bottom_names,top_name):
    kernel_h,kernel_w=config['kernel_size']
    num_output=config['out_channels']
    stride_h,stride_w=config['stride']
    dilation=config['dilation']
    group=config['groups']
    pad_h,pad_w=config['padding']
    bias_term='true' if config['use_bias'] else 'false'
    return create_layer('Convolution',layer_name,bottom_names,top_name,kernel_h=kernel_h,kernel_w=kernel_w,stride_h=stride_h,stride_w=stride_w,pad_h=pad_h,pad_w=pad_w,dilation=dilation[0],bias_term=bias_term,num_output=num_output,group=group,binary='true')


create_hiquantconv2d=create_conv2d
create_hiquantconvtranspose2d=create_convtranspose2d
create_hiquantmaxpool2d=create_maxpool2d
create_hiquantconcat=create_concat