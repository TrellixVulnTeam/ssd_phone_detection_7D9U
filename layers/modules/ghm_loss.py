
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GHMLoss(nn.Module):
    def __init__(self, bins=10, **kargs):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.edges = torch.arange(bins + 1).float() / bins

    def forward(self, inputs, target,reduction='sum'):
        pred = inputs.max(dim=1)[0]
        inputs_l = inputs.softmax(1).gather(1,target.unsqueeze(1)).squeeze(1)
        # 计算梯度模长
        g = torch.abs(inputs_l.detach() - target.type_as(inputs_l)).cpu()
        # 通过循环计算落入10个bins的梯度模长数量
        weights = torch.Tensor(pred.size())
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            print(i,num_in_bin)
            
            if num_in_bin > 0:
                # 重点，所谓的梯度密度就是1/num_in_bin
                weights[inds] = 1 / num_in_bin
        weights *= pred.size(0)/self.bins
        print(weights)
        exit(0)
        # 把上面计算的weights填到binary_cross_entropy_with_logits里就行了
        loss = F.cross_entropy(inputs, target, reduction='none') 
        # print(loss.size())
        # print(weights.size())
        loss *= weights.cuda()
        loss = loss.sum()
        return loss

# def _expand_binary_labels(labels, label_weights, label_channels):
#     bin_labels = labels.new_full((labels.size(0), label_channels), 0)
#     inds = torch.nonzero(labels >= 1).squeeze()
#     if inds.numel() > 0:
#         bin_labels[inds, labels[inds] - 1] = 1
#     bin_label_weights = label_weights.view(-1, 1).expand(
#         label_weights.size(0), label_channels)
#     return bin_labels, bin_label_weights

# class GHMC(nn.Module):
#     """GHM Classification Loss.
#     Details of the theorem can be viewed in the paper
#     "Gradient Harmonized Single-stage Detector".
#     https://arxiv.org/abs/1811.05181
#     Args:
#         bins (int): Number of the unit regions for distribution calculation.
#         momentum (float): The parameter for moving average.
#         use_sigmoid (bool): Can only be true for BCE based loss now.
#         loss_weight (float): The weight of the total GHM-C loss.
#     """
#     def __init__(
#             self,
#             bins=10,
#             momentum=0,
#             use_sigmoid=True,
#             loss_weight=1.0):
#         super(GHMC, self).__init__()
#         self.bins = bins
#         self.momentum = momentum
#         self.edges = torch.arange(bins + 1).float().cuda() / bins
#         self.edges[-1] += 1e-6
#         if momentum > 0:
#             self.acc_sum = torch.zeros(bins).cuda()
#         self.use_sigmoid = use_sigmoid
#         if not self.use_sigmoid:
#             raise NotImplementedError
#         self.loss_weight = loss_weight

#     def forward(self, pred, target, label_weight, *args, **kwargs):
#         """Calculate the GHM-C loss.
#         Args:
#             pred (float tensor of size [batch_num, class_num]):
#                 The direct prediction of classification fc layer.
#             target (float tensor of size [batch_num, class_num]):
#                 Binary class target for each sample.
#             label_weight (float tensor of size [batch_num, class_num]):
#                 the value is 1 if the sample is valid and 0 if ignored.
#         Returns:
#             The gradient harmonized loss.
#         """
#         # the target should be binary class label
#         if pred.dim() != target.dim():
#             target, label_weight = _expand_binary_labels(
#                                     target, label_weight, pred.size(-1))
#         target, label_weight = target.float(), label_weight.float()
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(pred)

#         # gradient length
#         g = torch.abs(pred.sigmoid().detach() - target)

#         valid = label_weight > 0
#         tot = max(valid.float().sum().item(), 1.0)
#         n = 0  # n valid bins
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1]) & valid
#             num_in_bin = inds.sum().item()
#             if num_in_bin > 0:
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                         + (1 - mmt) * num_in_bin
#                     weights[inds] = tot / self.acc_sum[i]
#                 else:
#                     weights[inds] = tot / num_in_bin
#                 n += 1
#         if n > 0:
#             weights = weights / n

#         loss = F.binary_cross_entropy_with_logits(
#             pred, target, weights, reduction='sum') / tot
#         return loss * self.loss_weight