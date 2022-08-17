# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from data import coco as cfg
from ..box_utils import match, log_sum_exp,jaccard,point_form,bbox_overlaps_giou,decode
# from . import focal_loss as focalLoss
# from . import ghm_loss as ghm_loss
from . import labe_smooth_loss as labe_smooth_loss

import glog
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns


def priorbox2featuremap(priorbox_class_stats):
    priorbox_class_stats=priorbox_class_stats.detach().cpu().numpy()
    np.save("priorbox_class_stats.npy",priorbox_class_stats)
    # save_stats(priorbox_class_stats[0],'car')
    # save_stats(priorbox_class_stats[1],'bus_truck')
    # save_stats(priorbox_class_stats[2],'pedestrian')

class IouLoss(nn.Module):

    def __init__(self,pred_mode = 'Center',size_sum=True,variances=None,losstype='Giou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype
    def forward(self, loc_p, loc_t,prior_data):
        num = loc_p.shape[0] 
        
        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p

        if self.loss == 'Giou':
            loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes,loc_t))         
     
        if self.size_sum:
            loss = loss
        else:
            loss = loss/num
        return loss

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    #aspect_ratios': 2+2*[[2], [2, 3], [2, 3], [2, 3], [2], [2],[2]]
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True,cross_dataset=True,use_atss=False,no_conflict_label=3):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance =[0.1,0.2]
        self.register_buffer("priorbox_class_stats",torch.ones(1,2046).cuda().long())
        self.cross_dataset=cross_dataset
        self.index=0
        self.use_atss = use_atss
        # self.focal_Loss = focalLoss.FocalLoss(class_num=2,gamma=2, alpha=0.75)
        self.LSR = labe_smooth_loss.LSR()
        self.gious = IouLoss(pred_mode = 'Center',size_sum=True,variances=self.variance, losstype='Giou')
        
        # self.ghm_loss = ghm_loss.GHMLoss()
        # self.ghm_loss = ghm_loss.GHMLoss()
        if self.cross_dataset:
            self.conflict_label_list=[]
            for i in range(self.num_classes):
                if i!=no_conflict_label:
                    self.conflict_label_list.append(i)

    def boxlist_iou(self,boxlist1, boxlist2):
        """Compute the intersection over union of two set of boxes.
        The box order must be (xmin, ymin, xmax, ymax).
        Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
        Returns:
        (tensor) iou, sized [N,M].
        Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        """
        if boxlist1.size != boxlist2.size:
            print(boxlist1.size,boxlist2.size)
            raise RuntimeError(
                    "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

        N = len(boxlist1)
        M = len(boxlist2)

        area1 = boxlist1.area()
        area2 = boxlist2.area()

        box1, box2 = boxlist1.bbox, boxlist2.bbox

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        TO_REMOVE = 1

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou

  
    def forward(self, predictions, targets,conflict_flag):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        

        loc_data, conf_data, priors = predictions


        num = loc_data.size(0)
        # glog.info("{} {} {}".format(len(conflict_flag),num,len(targets[0])))
        if num==0:
            return torch.zeros(1).cuda(),torch.zeros(1).cuda()
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)



      





        for idx in range(num):
            num_gt = targets[idx].size(0)
            
            truths = targets[idx][:, :-1].data#左上右下
            labels = targets[idx][:, -1].data#0,1

            defaults = priors.data#所有默认框
    
            # import glog
            # glog.info("{}".format(truths))
            # glog.info("{} {} {} {}".format(truths.shape,defaults.shape,loc_t.shape,conf_t.shape))
            if self.use_atss:
                defaults_ = point_form(defaults)
                ious = jaccard(truths,defaults_)
                
                gt_cx = (truths[:, 2] + truths[:, 0]) / 2.0
                gt_cy = (truths[:, 3] + truths[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)
                anchors_cx_per_im = (defaults_[:, 2] + defaults_[:, 0]) / 2.0
                anchors_cy_per_im = (defaults_[:, 3] + defaults_[:, 1]) / 2.0
                anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                candidate_idxs = []
                star_idx = 0
                # anchor_per_level=[0,1536,1920,2304,2400,2496]
                anchor_per_level=[0,1536,1920,2016,2040,2046]

                for i in range(len(anchor_per_level)-1):
                    start = anchor_per_level[i]
                    end = anchor_per_level[i+1]
                    _, topk_idxs_per_level = distances[start:end].topk(min(9*6,end-start),dim=0,largest=False)
                    candidate_idxs.append(topk_idxs_per_level+start)
                candidate_idxs = torch.cat(candidate_idxs, dim=0)
                candidate_idxs = candidate_idxs.permute(1,0)
      
            
                
                candidate_ious = torch.gather(ious,dim=1,index=candidate_idxs)
               
            
                iou_mean_per_gt = candidate_ious.mean(1)
                iou_std_per_gt = candidate_ious.std(1)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        
            
                
                # is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                match(iou_thresh_per_gt, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx,self.use_atss)
      
            else:
                match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx,self.use_atss)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)




        self.priorbox_class_stats[0].copy_(self.priorbox_class_stats[0]+torch.sum(conf_t==1,dim=0))
        # self.priorbox_class_stats[1].copy_(self.priorbox_class_stats[1]+torch.sum(conf_t==2,dim=0))
        # self.priorbox_class_stats[2].copy_(self.priorbox_class_stats[2]+torch.sum(conf_t==3,dim=0))
        if self.index%100==0:
            priorbox2featuremap(self.priorbox_class_stats)

        self.index=self.index+1

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        ##Giou loss
        giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
        loss_l = loss_l + self.gious(loc_p,loc_t,giou_priors[pos_idx].view(-1, 4))

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining

        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        
        # num_pos[num_pos==0] = 2
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)




        if self.cross_dataset:
            conflict_flag=torch.from_numpy(np.array(conflict_flag)).to(conf_data.device)
            conf_p = conf_data.view(-1, self.num_classes)#(pos_idx+neg_idx).gt(0)
            pp=F.softmax(conf_p, dim=1).log().view(num,-1, self.num_classes)
            pp_conflict=pp[conflict_flag==1]
            pp_conflict[:,self.conflict_label_list]=0
            pp[conflict_flag==1].copy_(pp_conflict)
            conf_p = pp[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos+neg).gt(0)]
            loss_c = F.nll_loss(conf_p,targets_weighted, size_average=False)
        else:
            conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos+neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
            # loss_c = self.LSR(conf_p, targets_weighted)


            # alpha = np.array([[0.25], [0.75], [0.75]])
            # alpha = torch.Tensor(alpha)
            # compute_c_loss = focalLoss.FocalLoss(alpha=alpha, gamma=2, class_num=self.num_classes, size_average=False)
            # loss_c =compute_c_loss(conf_p, targets_weighted)

            # loss_c = self.ghm_loss(conf_p, targets_weighted)

        # print(targets_weighted,targets_weighted.shape)



        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
  
        return loss_l, loss_c
        # return 0, loss_c
