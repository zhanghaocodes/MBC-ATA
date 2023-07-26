# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from torch.nn.modules.loss import _Loss
class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on   #False
        self.num_attri_cat = num_attri_cat  #201
        self.max_num_attri = max_num_attri  #10
        self.attribute_sampling = attribute_sampling  #True
        self.attribute_bgfg_ratio = attribute_bgfg_ratio  #3
        self.use_label_smoothing = use_label_smoothing   #False
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:  #False
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss =  -1 * (1-pt)**self.gamma* logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
def create_FocalLoss():
    print("Focal Loss is used by HAO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    return FocalLoss(gamma = 2.0)





class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq):
        super(BalancedSoftmax, self).__init__()

        self.sample_per_class = freq

    def forward(self, input, label, reduction='sum'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)
def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
def create_BalancedSoftmax():
    cls_num_list = [118037,118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507, 3808,
                    2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580, 512, 511,
                    493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]
    freq = torch.Tensor(cls_num_list)

    return BalancedSoftmax(freq)





def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()
class IBLoss(nn.Module):
    def __init__(self, weight=None,num_classes=0, alpha=10000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.num_classes=num_classes
    def forward(self, input, target, features):

        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)),1) # N * 1
        features=torch.sum(torch.abs(features), 1).reshape(-1, 1)
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)



def create_CBLoss():
    # cls_num_list = [118037,118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482,
    #                 8411, 5355, 4939, 4732, 4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277,
    #                 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580, 512, 511, 493, 485,
    #                 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]
    cls_num_list = [118037,118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507, 3808, 2496, 2109,
     1705, 1586, 1322, 1277, 1116, 1026, 907, 3228, 3116, 2808, 2716, 2608, 2564, 2320, 2048, 2044, 1972, 1940, 1740,
     1476, 1372, 1308, 1156, 1060, 1052, 896, 792, 688, 612, 536, 512, 196, 20]

    # beta = 0.9999
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

    per_cls_weights = per_cls_weights.tolist()


    per_cls_weights = torch.FloatTensor(per_cls_weights)
    result_str ="The Loss is [ CBLoss ]"
    result_str += ('\n   Weight is {}'.format(per_cls_weights))
    with open(('/home/share/zhanghao/data/image/datasets/output/LossInfo.txt'), 'w') as outfile:
        outfile.write(result_str)
    print("CBLoss is over @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    # return nn.CrossEntropyLoss(weight=per_cls_weights,reduction='sum')
    return  CE(per_cls_weights)


class SeesawLoss(nn.Module):
    def __init__(self, p: float = 0.8):
        super().__init__()
        dist = [118037,118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580, 512, 511, 493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]

        class_counts = torch.FloatTensor(dist)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        print(trues.dtype)
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.num_labels = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = F.one_hot(targets, self.num_labels)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
                              (1 - targets)[:, None, :]
                              * self.s[None, :, :]
                              * torch.exp(logits)[:, None, :]).sum(axis=-1) \
                      + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


def create_SeesawLoss():
    print("SeesawLoss is used by HAO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    return SeesawLoss()


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator


def create_WeightedSoftmaxLoss():
    cls_num_list = [118037,118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580, 512, 511, 493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]


    per_cls_weights = []
    for i in range(len(cls_num_list)):
        if (cls_num_list[i] > 8000):
            per_cls_weights.append((1 / cls_num_list[i]))
        elif (cls_num_list[i] > 3000):
            per_cls_weights.append((10 / cls_num_list[i]))
        elif (cls_num_list[i] > 1000):
            per_cls_weights.append((100 / cls_num_list[i]))
        else:
            per_cls_weights.append((200 / cls_num_list[i]))



    per_cls_weights = torch.FloatTensor(per_cls_weights)


    return nn.CrossEntropyLoss(weight=per_cls_weights,reduction='sum')



class CE(nn.Module):
    def __init__(self, weight=None):
        super(CE, self).__init__()

        self.lossfun = nn.CrossEntropyLoss(weight=weight,reduction="sum")
    def forward(self, input, target):
        # first = 0  # 第一项
        # second = 0  # 第二项
        # for i in range(target.size(0)):
        #     first += -input[i][target[i]] * self.weight[target[i]]
        #     tempSum = 0
        #     for j in range(input.size(1)):
        #         tempSum += torch.exp(input[i][j])
        #     second += torch.log(tempSum) * self.weight[target[i]]
        # res = (first + second)/target.size(0)

        res = self.lossfun(input,target)
        return res/input.size(0)


def create_CE():
    # cls_num_list = [118037, 118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732,
    #                 4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580,
    #                 512, 511, 493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]
    #
    #
    #
    # per_cls_weights = []
    #
    # for i in range(51):
    #     per_cls_weights.append(0)
    #
    #
    # sort_class_num = sorted(cls_num_list)
    # for i in range(len(cls_num_list)):
    #     if cls_num_list[i]>sort_class_num[25]:
    #         per_cls_weights[i] = sort_class_num[25]/cls_num_list[i]
    #     else:
    #         per_cls_weights[i] = 1





    # for i in range(len(cls_num_list)):
    #     if i<25:
    #         per_cls_weights.append(cls_num_list[25]/cls_num_list[i])
    #     else:
    #         per_cls_weights.append(1)

    # beta = 0.9999
    # effective_num = 1.0 - np.power(beta, cls_num_list)
    # per_cls_weights = (1.0 - beta) / np.array(effective_num)
    # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    #
    # per_cls_weights = per_cls_weights.tolist()
    per_cls_weights = [0.6836839296152901,0.6836839296152901, 1.1694465778834031, 1.661109052735581, 2.462618248397925, 3.2979158152840213,
               3.8874704947251795, 6.185329960910554, 6.606631191158412, 7.0283922661557225, 9.59457852811794,
               15.070028011204482, 16.339339947357765, 17.054099746407438, 17.90548036387841, 21.192226890756302,
               32.331730769230774, 38.26458036984353, 47.33137829912024, 50.88272383354351, 61.043872919818455,
               63.19498825371965, 72.31182795698925, 78.65497076023392, 88.97464167585446, 100.0, 103.59435173299102,
               114.95726495726495, 118.85125184094257, 123.77300613496934, 125.89703588143526, 139.13793103448276,
               157.6171875, 157.9256360078278, 163.69168356997972, 166.3917525773196, 185.51724137931035,
               218.6991869918699, 235.27696793002914, 246.78899082568807, 279.2387543252595, 304.52830188679246,
               306.84410646387835, 360.26785714285717, 407.5757575757576, 469.1860465116279, 527.4509803921569,
               602.2388059701493, 630.46875, 1646.938775510204, 16140.0]

    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return CE(per_cls_weights)


def choose_rel_loss():



    return create_CBLoss()
   #return nn.CrossEntropyLoss()

def create_50weighted_CE(per_cls_weights):
    # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return CE(per_cls_weights)
def getCB(cls_num_list):



    # beta = 0.9999
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)

    per_cls_weights = per_cls_weights.tolist()


    per_cls_weights_final = [10*i for i in per_cls_weights]

    # per_cls_weights = torch.FloatTensor(per_cls_weights)
    print(per_cls_weights_final)
    return per_cls_weights_final


def generate_50_rel_loss():
    loss_list = []
    relative_freq = [[6133407, 118037], [3411029, 69007], [3595348, 48582], [2180724, 32770], [2369140, 24470],
                     [1938859, 20759], [1393963, 13047], [1511129, 12215], [1279338, 11482], [688665, 8411],
                     [632023, 5355], [558007, 4939], [473400, 4732], [447523, 4507], [518156, 3808], [329784, 2496],
                     [236203, 2109], [251971, 1705], [159890, 1586], [213304, 1322], [190985, 1277], [141220, 1116],
                     [159188, 1026], [146525, 907], [128895, 807], [68861, 779], [74842, 702], [98873, 679],
                     [69744, 652], [115391, 641], [80374, 580], [86930, 512], [67299, 511], [83683, 493],
                     [73989, 485], [35369, 435], [35431, 369], [50683, 343], [53997, 327], [52429, 289],
                     [31145, 265], [42665, 263], [34832, 224], [29518, 198], [24860, 172], [19633, 153],
                     [10826, 134], [11308, 128], [4873, 49], [849, 5]]


    for i in range(50):
        cls = relative_freq[i]
        weight = getCB(cls)
        per_cls_weights = torch.FloatTensor(weight).cuda()
        loss_list.append(create_50weighted_CE(per_cls_weights))
    print("********************************************generate_50_rel_loss()**************************************************************")
    return loss_list

def generate_50_rel_loss_ata():
    print("********************************************generate_50_rel_loss_ata()**************************************************************")
    loss_list = []


    relative_freq = [[6453647, 118037], [3569591, 69007], [3756152, 48582], [2299644, 32770], [2489688, 24470],
                     [2054449, 20759], [1467731, 13047], [1589591, 12215], [1341918, 11482], [734735, 8411],
                     [671127, 5355], [577805, 4939], [495768, 4732], [469355, 4507], [548892, 3808], [353488, 2496],
                     [247949, 2109], [265297, 1705], [174628, 1586], [233332, 1322], [202785, 1277], [149286, 1116],
                     [169856, 1026], [151775, 907], [176754, 3228], [97558, 3116], [101754, 2808], [133990, 2716],
                     [109562, 2608], [158544, 2564], [108438, 2320], [121002, 2048], [93264, 2044], [115540, 1972],
                     [103586, 1940], [56202, 1740], [51530, 1476], [68860, 1372], [73090, 1308], [70854, 1156],
                     [46904, 1060], [58972, 1052], [47676, 896], [40336, 792], [38420, 688], [28842, 612], [15852, 536],
                     [16554, 512], [6762, 196], [1102, 20]]


    for i in range(50):
        cls = relative_freq[i]
        weight = getCB(cls)
        per_cls_weights = torch.FloatTensor(weight).cuda()
        loss_list.append(create_50weighted_CE(per_cls_weights))
    return loss_list




from torch.nn.modules.loss import _WeightedLoss

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight = None, reduction = 'mean', smoothing = 0.1, pos_weight = None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing = 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad(): targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight, pos_weight = self.pos_weight)
        if  self.reduction == 'sum': loss = loss.sum()
        elif  self.reduction == 'mean': loss = loss.mean()
        return loss
