import torch.nn as nn
import torch.nn.functional as F
import torch
from custom_nn import SinkhornDistance

import math
class WassersteinOrthogonal(nn.Module):
    def __init__(self,in_channels,out_channels,beta=(1.0,1.0,0.001),alpha=(1.0,1.0),R_term=True,distribution_criterion=SinkhornDistance( 0.00001, 10000, reduction='mean').cuda()):
        super(WassersteinOrthogonal,self).__init__()
        self.register_parameter("shared_weight",torch.nn.Parameter(torch.ones(out_channels,in_channels,1,1)))
        self.register_parameter("p_weight",torch.nn.Parameter(torch.ones(out_channels,in_channels,1,1)))
        self.register_parameter("q_weight",torch.nn.Parameter(torch.ones(out_channels,in_channels,1,1)))
        self.distribution_criterion=distribution_criterion
        self.shared_weight.data.uniform_(-1.0/math.sqrt(in_channels),1.0/math.sqrt(in_channels))
        self.p_weight.data.uniform_(-1.0/math.sqrt(in_channels),1.0/math.sqrt(in_channels))
        self.q_weight.data.uniform_(-1.0/math.sqrt(in_channels),1.0/math.sqrt(in_channels))
        self.alpha=alpha
        self.beta=beta
        self.R_term=R_term
    def forward(self,x,y):
        x_p = F.conv2d(input=x, weight=self.p_weight, stride=(1, 1), groups=1)
        y_q = F.conv2d(input=y, weight=self.q_weight, stride=(1, 1), groups=1)
        x_shared=F.conv2d(input=x, weight=self.shared_weight, stride=(1, 1), groups=1)
        y_shared=F.conv2d(input=y, weight=self.shared_weight, stride=(1, 1), groups=1)
        x=x_p+x_shared
        y=y_q+y_shared
        

        shared_weight=self.shared_weight.view(self.shared_weight.size(0),-1)#o,i
        p_weight=self.p_weight.view(self.p_weight.size(0),-1).transpose(1,0)#i,o
        q_weight=self.q_weight.view(self.q_weight.size(0),-1).transpose(1,0)#i,o
        print(shared_weight.shape)
        l2_regulation=shared_weight.matmul(p_weight).pow(2).sum().sqrt()*self.alpha[0]+shared_weight.matmul(q_weight).pow(2).sum().sqrt()*self.alpha[1]

        x_shared=x_shared.view(x_shared.size(0),x_shared.size(1),-1).permute(0,2,1).contiguous()
        y_shared=x_shared.view(y_shared.size(0),y_shared.size(1),-1).permute(0,2,1).contiguous()
        cost, pi, C=self.distribution_criterion(x_shared,y_shared)

        loss=cost*self.beta[1]+l2_regulation

        if self.R_term:
            M=torch.cat([x.view(x_shared.size(0),x_shared.size(1),-1),y.view(y_shared.size(0),y_shared.size(1),-1)],dim=1).permute(0,2,1).contiguous()
            M=M.view(-1,M.size(2))
            U,S,V=torch.svd(M)
            R=S.pow(2).sqrt().sum()
            loss=R*self.beta[2]+loss
        
        return x,y,loss

if __name__=='__main__':
    model=WassersteinOrthogonal(512,256,R_term=False).cuda()
    model(torch.rand(2,512,32,32).cuda(),torch.rand(2,512,32,32).cuda())