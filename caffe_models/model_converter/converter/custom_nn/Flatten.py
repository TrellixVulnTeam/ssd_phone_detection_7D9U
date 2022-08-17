import torch
import torch.nn as nn
import torch.nn.functional as F
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        pass
    def forward(self,*inputs):
        if not inputs[0].is_contiguous():
            output=inputs[0].contiguous().view(inputs[0].size(0),-1)
        else:
            output=inputs[0].view(inputs[0].size(0),-1)
        return output