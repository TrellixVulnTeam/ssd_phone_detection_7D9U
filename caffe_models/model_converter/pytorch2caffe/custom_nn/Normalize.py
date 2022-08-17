import torch
import torch.nn as nn
import torch.nn.functional as F
class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    def forward(self,x):
        #print(x.shape)
        return F.normalize(x,p=2,dim=1,eps=1e-12)