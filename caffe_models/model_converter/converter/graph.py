from collections import OrderedDict
import torch.nn as nn
import numpy as np
import os

def layer_rename(layer_name):
    layer_name=layer_name.replace(".","_").replace('/','_')
    if ":" in layer_name:
        return layer_name.split(":")[0]
    else:
        return layer_name

def build_model_graph(layers,graph,hooks,prefix='',feature_dir=None):
    keys=layers._modules.keys()
    for i,key in enumerate(keys):
        layer_=layers._modules[key]
        layer_.layer_name=(prefix+'.'+key) if prefix!='' else ''+key
        layer_.feature_dir=feature_dir
        def get_shape(module,input,output):
            m_key=module.layer_name
            graph[m_key]=dict()

            vars_=vars(module)
            vars_new=dict([(var,vars_[var]) for var in vars_ if var not in ["_buffers","_parameters","_backend","training","_modules","layer_name"] and "hooks" not in var])
            vars_new['use_bias']=hasattr(module,'bias')
            vars_new['use_weight']=hasattr(module,'weight')
            graph[m_key]['layer_type']=str(module.__class__).split(".")[-1].split("'")[0]
            graph[m_key]["config"] =vars_new
            def get_input_info(data):
                if hasattr(data,'id_name'):
                    return (data.id_name,tuple(data.shape))
                else:
                    return (str(id(data)),tuple(data.shape))
            graph[m_key]["config"]["bottoms"] =tuple(map(get_input_info,input))


            def add_output_info(idx_data):
                idx,data=idx_data
                data.id_name="{}:{}".format(m_key,idx)
                return tuple(data.shape)

            if isinstance(output, (list, tuple)):
                tuple(map(add_output_info,enumerate(output)))
            else:
                add_output_info((0,output))
                def save_featuremap(data):
                    if module.feature_dir is not None and os.path.isdir(module.feature_dir):
                        np.save(os.path.join(module.feature_dir,"{}.npy".format(layer_rename(data.id_name))),data.detach().cpu().numpy())
                save_featuremap(output)
        if (
            not isinstance(layer_, nn.Sequential)
            and not isinstance(layer_, nn.ModuleList)
            and not (layer_ == layers)
        ):
            hooks.append(layer_.register_forward_hook(get_shape))
        if len(layer_._modules.keys())>0:
            build_model_graph(layer_,graph,hooks=hooks,prefix=layer_.layer_name,feature_dir=feature_dir)