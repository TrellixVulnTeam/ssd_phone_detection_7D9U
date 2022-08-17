import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_nn as custom_nn
class CosineLinear(nn.Module):
    def __init__(self,
                    in_feats,
                    n_classes=10516
                    ):
        super(CosineLinear, self).__init__()
        self.in_feats = in_feats#you need normalize embedding kernel
        self.n_classes =n_classes
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_feats), requires_grad=True)
        # self.weights = torch.nn.Parameter(torch.randn(in_feats,n_classes), requires_grad=True)
        # nn.init.xavier_normal_(self.weights.data)#, gain=1
        nn.init.xavier_uniform_(self.weight)



    def forward(self, x):
        assert x.size()[1] == self.in_feats
        # with torch.no_grad():
        weights_norm = F.normalize(self.weight, p=2, dim=1,eps=1e-12)#.clamp(min=-1,max=1)
        costh = F.linear(x, weights_norm)
        return costh
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_feats=' + str(self.in_feats) \
            + ', n_classes=' + str(self.n_classes)+ ')'