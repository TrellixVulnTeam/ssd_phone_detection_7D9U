import torch
import torch.nn as nn
import torch.nn.functional as F
class Softmax(nn.Module):
    def __init__(self,axis):
        super(Softmax, self).__init__()
        self.axis=axis
    def forward(self,*inputs):
        output=F.softmax(inputs[0],dim =self.axis)
        return output