import torch
import torch.nn as nn
import torch.nn.functional as F
class Permute(nn.Module):
    def __init__(self,order):
        super(Permute, self).__init__()
        self.order=order
    def forward(self,*inputs):
        output=inputs[0].permute(*self.order)
        return output