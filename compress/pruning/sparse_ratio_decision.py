import subprocess
import glog
import numpy as np
def sparse_ratio_decision(weights,energy_acc,energy_all,energy, stride_group, max_layer_group, layer_name, ctrl_bits, is_dw, prun_algo, prun_algo_tuning, bin_core_sparse_ratio_decision):
    # based on energy and layer group to derive final energy for each layer
    # after getting final energy, each layer can get pruning ratio
    # return pruning ratio for each layer
    sparse_ratio = 0.0
    energy_th = energy
    layer_param = 0
    layer_group = max_layer_group - stride_group
    MAX_SPARSE=0.99

    if len(weights.shape)==4: # convolution
        scale_dw = 2 if is_dw else 1
        if prun_algo==-1:
            energy_th = energy_th # this case consider energy_th as pruning ratio directly, instead of energy
        elif prun_algo==0:
            energy_th = energy_th*(1-min(0.9, layer_group*scale_dw*20/100))
        else:
            argc_list='{} {} {} {} {} {} {}'.format(prun_algo, prun_algo_tuning, energy_th, stride_group, max_layer_group, ctrl_bits, is_dw)
            result=subprocess.Popen('{} {}'.format(bin_core_sparse_ratio_decision, argc_list), shell=True, stdout=subprocess.PIPE).communicate()[0]
            energy_th = float(result)
        out_c, in_c, w, h = weights.shape
        layer_param = out_c*in_c*w*h
    elif len(weights.shape)==2: # fully connected
        out_c, in_c = weights.shape
        layer_param = out_c*in_c
    if prun_algo>0:
        # use energy as criterion
        # sorted_data = weights.abs().view(-1).sort()[0]
        # square_data=sorted_data.pow(2)
        # energy_all = square_data.sum()
        # # accumulate the energy to get sparse ratio
        # energy_acc=square_data.cumsum( dim=-1)
        energy_ratio_masks=~((energy_acc/energy_all)*100>energy_th)
        sparse_ratio=energy_ratio_masks.sum()#+1
        sparse_ratio = int(100.0*sparse_ratio/layer_param+0.5)/100.0
    else:
        sparse_ratio = energy_th
    # use 1.0 will let some network directly die so avoid from using 1.0 for pruning ratio so add maximum value MAX_SPARSE
    sparse_ratio = min(sparse_ratio, MAX_SPARSE)

    return sparse_ratio