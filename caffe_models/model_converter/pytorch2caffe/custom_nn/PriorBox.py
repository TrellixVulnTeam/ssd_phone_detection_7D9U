import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def corner2center(corner_priorbox):
    xy_center=(corner_priorbox[:,[0,1]]+corner_priorbox[:,[2,3],])/2.0
    wh=corner_priorbox[:,[2,3],]-corner_priorbox[:,[0,1]]
    anchors=np.concatenate([xy_center,wh],axis=1)
    return anchors
def center2corner(center_priorbox):
    xy_center=center_priorbox[:,:2]
    wh=center_priorbox[:,2:]
    anchors=np.concatenate([xy_center-wh/2,xy_center+wh/2],axis=1)
    return anchors
class PriorBox(nn.Module):
    def __init__(self,npy_path='ssd512_mbox_priorbox.pth.npy'):
        super(PriorBox, self).__init__()
        center_priorbox=np.load(npy_path)#[0,0,:].reshape(-1,4)
        # center_priorbox=corner2center(corner_priorbox)
        self.register_buffer('center_priorbox',torch.from_numpy(center_priorbox))
    def forward(self):
        return self.center_priorbox
