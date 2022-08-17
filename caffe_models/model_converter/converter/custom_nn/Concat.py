import torch
import torch.nn as nn
import torch.nn.functional as F
class Concat(nn.Module):
    def __init__(self,axis=1,n_inputs=2):
        super(Concat, self).__init__()
        self.axis=axis
        self.n_inputs=n_inputs
    def forward(self,*inputs):
        output=torch.cat(inputs,dim=self.axis)
        return output