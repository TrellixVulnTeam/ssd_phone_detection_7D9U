import torch
import torch.nn as nn
import torch.nn.functional as F
class Reshape(nn.Module):
    def __init__(self,shape):
        super(Reshape, self).__init__()
        assert shape[0]==0
        self.shape=shape

    def forward(self,*inputs):
        output=inputs[0].view(inputs[0].size(0),*self.shape[1:])
        return output