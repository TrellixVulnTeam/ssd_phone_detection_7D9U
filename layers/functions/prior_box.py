from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
import numpy as np

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        self.crop_sizes=cfg['crop_sizes']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []

        mean_list=[]

        for k, f in enumerate(self.feature_maps):
            # print(k,f,len(mean_list)/4)
            x_pos=np.linspace(f//2-self.crop_sizes[k]//2,f//2+self.crop_sizes[k]//2-1,self.crop_sizes[k])
            y_pos=np.linspace(f//2-self.crop_sizes[k]//2,f//2+self.crop_sizes[k]//2-1,self.crop_sizes[k])
            X_pos,Y_pos=np.meshgrid(x_pos,y_pos)

            for i,j in np.stack([Y_pos,X_pos],axis=0).reshape(2,-1).T:
                print(f)
                f_k = self.image_size / self.steps[k]

                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k


                s_k = self.min_sizes[k]/self.image_size

                mean_list += [cx, cy, s_k, s_k]
                print(s_k,s_k)

                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean_list += [cx, cy, s_k_prime, s_k_prime]
                print(s_k_prime,s_k_prime)
                # hhhh+=4
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean_list += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    print( s_k*sqrt(ar),s_k/sqrt(ar))

                    mean_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                    print( s_k/sqrt(ar),s_k*sqrt(ar))
                # exit(0)


        output = torch.Tensor(mean_list).view(-1, 4)
        # exit(0)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
