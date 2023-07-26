# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import operator
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from SHA_GCL_extra.kl_divergence import KL_divergence
from .model_Hybrid_Attention import SHA_Context
from .model_Cross_Attention import CA_Context
from .utils_relation import layer_init

from maskrcnn_benchmark.data import get_dataset_statistics
from SHA_GCL_extra.utils_funcion import FrequencyBias_GCL, FrequencyBias_Group
from SHA_GCL_extra.extra_function_utils import generate_num_stage_vector, generate_sample_rate_vector, \
    generate_current_sequence_for_bias, get_current_predicate_idx
from SHA_GCL_extra.group_chosen_function import get_group_splits
import random
import torch.nn.functional as F

import math
from torch.nn import Parameter
def filter_rel_label_by_group(target_rel_labels):
    resultList = []
    # maxnum_20000 = [3, 4, 6, 15]
    # between6000and11000_8000 = [1, 2, 5, 12, 22, 48]
    # between1500and6000_3500 = [7, 17, 21, 23, 24, 37, 47, 49]
    # between500and1500_1000 = [13, 16, 18, 27, 38, 46, 50, 52, 55, 56]
    # between100and500 = [8, 10, 11, 14, 20, 28, 30, 31, 33, 40, 43, 44, 45]
    # under100_all = [9, 19, 25, 26, 29, 32, 34, 35, 36, 39, 41, 42, 51, 53, 54]

    maxnum_20000 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    between6000and11000_8000 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    between1500and6000_3500 = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    between500and1500_1000 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    between100and500 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]



    pred = [maxnum_20000,between6000and11000_8000,between1500and6000_3500,between500and1500_1000,between100and500]



    for i in range(5):
        target_rel_labels_res = target_rel_labels.clone()
        for rel in range(len(target_rel_labels_res)):
            if(pred[i].__contains__(target_rel_labels_res[rel])):
                target_rel_labels_res[rel] = pred[i].index(target_rel_labels_res[rel])+1
            else:
                target_rel_labels_res[rel] = 0
        resultList.append(target_rel_labels_res)
    return resultList
def filter_scores(scores,index):
    maxnum_20000 =  [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    between6000and11000_8000 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    between1500and6000_3500 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    between500and1500_1000 =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    between100and500 =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    pred = [maxnum_20000, between6000and11000_8000, between1500and6000_3500, between500and1500_1000, between100and500]

    tar_get_list = torch.tensor(pred[index])

    device =scores.device
    tar_get_list = tar_get_list.to(device)
    scores.index_fill_(1,tar_get_list,0)
    return scores







@registry.ROI_RELATION_PREDICTOR.register("TransLike_MBC")
class TransLike_MBC(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_MBC, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs

        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks()
        if (config.ATA):
            self.CE_loss = generate_50_rel_loss_ata()
        else:
            self.CE_loss = generate_50_rel_loss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics)
        # if self.Knowledge_Transfer_Mode != 'None':
        #     self.NLL_Loss = nn.NLLLoss()
        #     self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
        #     self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()
        '''
        torch.int64
        torch.float16
        '''

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        '''begin to change'''
        add_losses = {}

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            # group_target_label = filter_rel_label_by_group(rel_labels)



            mask_zero = torch.zeros_like(rel_labels)
            mask_one = torch.ones_like(rel_labels)

            for i in range(50):
                # rel_compress_test = self.rel_compress_all[i]
                # ctx_compress_test = self.ctx_compress_all[i]
                self.rel_compress_all[i].weight.requires_grad = True
                self.ctx_compress_all[i].weight.requires_grad = True
                rel_dists = self.rel_compress_all[i](visual_rep) + self.ctx_compress_all[i](prod_rep)


                tar_rel_label_group = torch.where(rel_labels==(i+1),mask_one,mask_zero)
                sum_rel = tar_rel_label_group.sum()
                device = rel_dists.device
                if(sum_rel == 0):
                    add_losses['%d_CE_loss' % (i + 1)] = torch.tensor([0]).float().to(device)
                    self.rel_compress_all[i].weight.requires_grad = False
                    self.ctx_compress_all[i].weight.requires_grad = False
                else:
                   if self.use_bias:
                       rel_bias_now = self.freq_bias_all[i]
                       rel_dists = rel_dists+rel_bias_now.index_with_labels(pair_pred.long())
                   add_losses['%d_CE_loss' % (i + 1)] = self.CE_loss[i](rel_dists, tar_rel_label_group)




            return None, None, add_losses

        else:
            obj_dists = obj_dists.split(num_objs, dim=0)

            rel_dists_sum = None
            for i in range(50):
                rel_dists = self.rel_compress_all[i](visual_rep) + self.ctx_compress_all[i](prod_rep)
                # rel_scores_tmp = rel_dists.split(num_rels, dim=0)
                # rel_scores_tmp = filter_scores(rel_dists, i)
                if self.use_bias:
                    rel_bias_test = self.freq_bias_all[i]
                    rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())


                rel_scores_tmp = F.softmax(rel_dists, -1)



                if (rel_dists_sum is None):
                    rel_dists_sum =rel_scores_tmp
                else:
                    # rel_dists_sum += rel_scores_tmp
                    # print(rel_dists_sum.size())
                    # print(rel_scores_tmp[:, 1:].size())
                    # print(rel_scores_tmp[:,-1].size())
                    rel_dists_sum = torch.cat((rel_dists_sum,rel_scores_tmp[:,1:]),1)

            rel_dists_result = rel_dists_sum.split(num_rels, dim=0)

            return obj_dists, rel_dists_result, add_losses





    def generate_muti_networks(self):
        classifer_all = []
        compress_all = []

        self.rel_classifer_1 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_1, xavier=True)
        classifer_all.append(self.rel_classifer_1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_1, xavier=True)
        compress_all.append(self.rel_compress_1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_2, xavier=True)
        classifer_all.append(self.rel_classifer_2)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_2, xavier=True)
        compress_all.append(self.rel_compress_2)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_3, xavier=True)
        classifer_all.append(self.rel_classifer_3)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_3, xavier=True)
        compress_all.append(self.rel_compress_3)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_4, xavier=True)
        classifer_all.append(self.rel_classifer_4)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_4, xavier=True)
        compress_all.append(self.rel_compress_4)
        self.rel_classifer_5 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_5, xavier=True)
        classifer_all.append(self.rel_classifer_5)
        self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_5, xavier=True)
        compress_all.append(self.rel_compress_5)
        self.rel_classifer_6 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_6, xavier=True)
        classifer_all.append(self.rel_classifer_6)
        self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_6, xavier=True)
        compress_all.append(self.rel_compress_6)
        self.rel_classifer_7 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_7, xavier=True)
        classifer_all.append(self.rel_classifer_7)
        self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_7, xavier=True)
        compress_all.append(self.rel_compress_7)
        self.rel_classifer_8 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_8, xavier=True)
        classifer_all.append(self.rel_classifer_8)
        self.rel_compress_8 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_8, xavier=True)
        compress_all.append(self.rel_compress_8)
        self.rel_classifer_9 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_9, xavier=True)
        classifer_all.append(self.rel_classifer_9)
        self.rel_compress_9 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_9, xavier=True)
        compress_all.append(self.rel_compress_9)
        self.rel_classifer_10 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_10, xavier=True)
        classifer_all.append(self.rel_classifer_10)
        self.rel_compress_10 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_10, xavier=True)
        compress_all.append(self.rel_compress_10)
        self.rel_classifer_11 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_11, xavier=True)
        classifer_all.append(self.rel_classifer_11)
        self.rel_compress_11 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_11, xavier=True)
        compress_all.append(self.rel_compress_11)
        self.rel_classifer_12 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_12, xavier=True)
        classifer_all.append(self.rel_classifer_12)
        self.rel_compress_12 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_12, xavier=True)
        compress_all.append(self.rel_compress_12)
        self.rel_classifer_13 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_13, xavier=True)
        classifer_all.append(self.rel_classifer_13)
        self.rel_compress_13 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_13, xavier=True)
        compress_all.append(self.rel_compress_13)
        self.rel_classifer_14 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_14, xavier=True)
        classifer_all.append(self.rel_classifer_14)
        self.rel_compress_14 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_14, xavier=True)
        compress_all.append(self.rel_compress_14)
        self.rel_classifer_15 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_15, xavier=True)
        classifer_all.append(self.rel_classifer_15)
        self.rel_compress_15 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_15, xavier=True)
        compress_all.append(self.rel_compress_15)
        self.rel_classifer_16 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_16, xavier=True)
        classifer_all.append(self.rel_classifer_16)
        self.rel_compress_16 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_16, xavier=True)
        compress_all.append(self.rel_compress_16)
        self.rel_classifer_17 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_17, xavier=True)
        classifer_all.append(self.rel_classifer_17)
        self.rel_compress_17 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_17, xavier=True)
        compress_all.append(self.rel_compress_17)
        self.rel_classifer_18 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_18, xavier=True)
        classifer_all.append(self.rel_classifer_18)
        self.rel_compress_18 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_18, xavier=True)
        compress_all.append(self.rel_compress_18)
        self.rel_classifer_19 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_19, xavier=True)
        classifer_all.append(self.rel_classifer_19)
        self.rel_compress_19 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_19, xavier=True)
        compress_all.append(self.rel_compress_19)
        self.rel_classifer_20 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_20, xavier=True)
        classifer_all.append(self.rel_classifer_20)
        self.rel_compress_20 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_20, xavier=True)
        compress_all.append(self.rel_compress_20)
        self.rel_classifer_21 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_21, xavier=True)
        classifer_all.append(self.rel_classifer_21)
        self.rel_compress_21 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_21, xavier=True)
        compress_all.append(self.rel_compress_21)
        self.rel_classifer_22 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_22, xavier=True)
        classifer_all.append(self.rel_classifer_22)
        self.rel_compress_22 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_22, xavier=True)
        compress_all.append(self.rel_compress_22)
        self.rel_classifer_23 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_23, xavier=True)
        classifer_all.append(self.rel_classifer_23)
        self.rel_compress_23 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_23, xavier=True)
        compress_all.append(self.rel_compress_23)
        self.rel_classifer_24 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_24, xavier=True)
        classifer_all.append(self.rel_classifer_24)
        self.rel_compress_24 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_24, xavier=True)
        compress_all.append(self.rel_compress_24)
        self.rel_classifer_25 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_25, xavier=True)
        classifer_all.append(self.rel_classifer_25)
        self.rel_compress_25 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_25, xavier=True)
        compress_all.append(self.rel_compress_25)
        self.rel_classifer_26 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_26, xavier=True)
        classifer_all.append(self.rel_classifer_26)
        self.rel_compress_26 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_26, xavier=True)
        compress_all.append(self.rel_compress_26)
        self.rel_classifer_27 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_27, xavier=True)
        classifer_all.append(self.rel_classifer_27)
        self.rel_compress_27 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_27, xavier=True)
        compress_all.append(self.rel_compress_27)
        self.rel_classifer_28 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_28, xavier=True)
        classifer_all.append(self.rel_classifer_28)
        self.rel_compress_28 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_28, xavier=True)
        compress_all.append(self.rel_compress_28)
        self.rel_classifer_29 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_29, xavier=True)
        classifer_all.append(self.rel_classifer_29)
        self.rel_compress_29 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_29, xavier=True)
        compress_all.append(self.rel_compress_29)
        self.rel_classifer_30 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_30, xavier=True)
        classifer_all.append(self.rel_classifer_30)
        self.rel_compress_30 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_30, xavier=True)
        compress_all.append(self.rel_compress_30)
        self.rel_classifer_31 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_31, xavier=True)
        classifer_all.append(self.rel_classifer_31)
        self.rel_compress_31 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_31, xavier=True)
        compress_all.append(self.rel_compress_31)
        self.rel_classifer_32 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_32, xavier=True)
        classifer_all.append(self.rel_classifer_32)
        self.rel_compress_32 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_32, xavier=True)
        compress_all.append(self.rel_compress_32)
        self.rel_classifer_33 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_33, xavier=True)
        classifer_all.append(self.rel_classifer_33)
        self.rel_compress_33 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_33, xavier=True)
        compress_all.append(self.rel_compress_33)
        self.rel_classifer_34 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_34, xavier=True)
        classifer_all.append(self.rel_classifer_34)
        self.rel_compress_34 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_34, xavier=True)
        compress_all.append(self.rel_compress_34)
        self.rel_classifer_35 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_35, xavier=True)
        classifer_all.append(self.rel_classifer_35)
        self.rel_compress_35 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_35, xavier=True)
        compress_all.append(self.rel_compress_35)
        self.rel_classifer_36 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_36, xavier=True)
        classifer_all.append(self.rel_classifer_36)
        self.rel_compress_36 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_36, xavier=True)
        compress_all.append(self.rel_compress_36)
        self.rel_classifer_37 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_37, xavier=True)
        classifer_all.append(self.rel_classifer_37)
        self.rel_compress_37 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_37, xavier=True)
        compress_all.append(self.rel_compress_37)
        self.rel_classifer_38 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_38, xavier=True)
        classifer_all.append(self.rel_classifer_38)
        self.rel_compress_38 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_38, xavier=True)
        compress_all.append(self.rel_compress_38)
        self.rel_classifer_39 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_39, xavier=True)
        classifer_all.append(self.rel_classifer_39)
        self.rel_compress_39 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_39, xavier=True)
        compress_all.append(self.rel_compress_39)
        self.rel_classifer_40 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_40, xavier=True)
        classifer_all.append(self.rel_classifer_40)
        self.rel_compress_40 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_40, xavier=True)
        compress_all.append(self.rel_compress_40)
        self.rel_classifer_41 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_41, xavier=True)
        classifer_all.append(self.rel_classifer_41)
        self.rel_compress_41 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_41, xavier=True)
        compress_all.append(self.rel_compress_41)
        self.rel_classifer_42 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_42, xavier=True)
        classifer_all.append(self.rel_classifer_42)
        self.rel_compress_42 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_42, xavier=True)
        compress_all.append(self.rel_compress_42)
        self.rel_classifer_43 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_43, xavier=True)
        classifer_all.append(self.rel_classifer_43)
        self.rel_compress_43 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_43, xavier=True)
        compress_all.append(self.rel_compress_43)
        self.rel_classifer_44 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_44, xavier=True)
        classifer_all.append(self.rel_classifer_44)
        self.rel_compress_44 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_44, xavier=True)
        compress_all.append(self.rel_compress_44)
        self.rel_classifer_45 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_45, xavier=True)
        classifer_all.append(self.rel_classifer_45)
        self.rel_compress_45 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_45, xavier=True)
        compress_all.append(self.rel_compress_45)
        self.rel_classifer_46 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_46, xavier=True)
        classifer_all.append(self.rel_classifer_46)
        self.rel_compress_46 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_46, xavier=True)
        compress_all.append(self.rel_compress_46)
        self.rel_classifer_47 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_47, xavier=True)
        classifer_all.append(self.rel_classifer_47)
        self.rel_compress_47 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_47, xavier=True)
        compress_all.append(self.rel_compress_47)
        self.rel_classifer_48 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_48, xavier=True)
        classifer_all.append(self.rel_classifer_48)
        self.rel_compress_48 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_48, xavier=True)
        compress_all.append(self.rel_compress_48)
        self.rel_classifer_49 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_49, xavier=True)
        classifer_all.append(self.rel_classifer_49)
        self.rel_compress_49 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_49, xavier=True)
        compress_all.append(self.rel_compress_49)
        self.rel_classifer_50 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_50, xavier=True)
        classifer_all.append(self.rel_classifer_50)
        self.rel_compress_50 = nn.Linear(self.hidden_dim * 2, 2)
        layer_init(self.rel_compress_50, xavier=True)
        compress_all.append(self.rel_compress_50)


        # for i in range(50):
        #     rel_classifer = nn.Linear(self.pooling_dim, 2)
        #     layer_init(rel_classifer, xavier=True)
        #     rel_compress = nn.Linear(self.hidden_dim * 2, 2)
        #     layer_init(rel_compress, xavier=True)
        #     classifer_all.append(rel_classifer)
        #     compress_all.append(rel_compress)
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics):
        freq_bias_all = []
        predicate_all_list = []

        for i in range(50):
            list_tmp = [0 for j in range(51)]
            list_tmp[i + 1] = i + 1
            predicate_all_list.append(list_tmp)

        self.freq_bias_1 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[0])
        freq_bias_all.append(self.freq_bias_1)
        self.freq_bias_2 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[1])
        freq_bias_all.append(self.freq_bias_2)
        self.freq_bias_3 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[2])
        freq_bias_all.append(self.freq_bias_3)
        self.freq_bias_4 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[3])
        freq_bias_all.append(self.freq_bias_4)
        self.freq_bias_5 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[4])
        freq_bias_all.append(self.freq_bias_5)
        self.freq_bias_6 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[5])
        freq_bias_all.append(self.freq_bias_6)
        self.freq_bias_7 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[6])
        freq_bias_all.append(self.freq_bias_7)
        self.freq_bias_8 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[7])
        freq_bias_all.append(self.freq_bias_8)
        self.freq_bias_9 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[8])
        freq_bias_all.append(self.freq_bias_9)
        self.freq_bias_10 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[9])
        freq_bias_all.append(self.freq_bias_10)
        self.freq_bias_11 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[10])
        freq_bias_all.append(self.freq_bias_11)
        self.freq_bias_12 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[11])
        freq_bias_all.append(self.freq_bias_12)
        self.freq_bias_13 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[12])
        freq_bias_all.append(self.freq_bias_13)
        self.freq_bias_14 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[13])
        freq_bias_all.append(self.freq_bias_14)
        self.freq_bias_15 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[14])
        freq_bias_all.append(self.freq_bias_15)
        self.freq_bias_16 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[15])
        freq_bias_all.append(self.freq_bias_16)
        self.freq_bias_17 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[16])
        freq_bias_all.append(self.freq_bias_17)
        self.freq_bias_18 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[17])
        freq_bias_all.append(self.freq_bias_18)
        self.freq_bias_19 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[18])
        freq_bias_all.append(self.freq_bias_19)
        self.freq_bias_20 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[19])
        freq_bias_all.append(self.freq_bias_20)
        self.freq_bias_21 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[20])
        freq_bias_all.append(self.freq_bias_21)
        self.freq_bias_22 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[21])
        freq_bias_all.append(self.freq_bias_22)
        self.freq_bias_23 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[22])
        freq_bias_all.append(self.freq_bias_23)
        self.freq_bias_24 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[23])
        freq_bias_all.append(self.freq_bias_24)
        self.freq_bias_25 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[24])
        freq_bias_all.append(self.freq_bias_25)
        self.freq_bias_26 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[25])
        freq_bias_all.append(self.freq_bias_26)
        self.freq_bias_27 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[26])
        freq_bias_all.append(self.freq_bias_27)
        self.freq_bias_28 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[27])
        freq_bias_all.append(self.freq_bias_28)
        self.freq_bias_29 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[28])
        freq_bias_all.append(self.freq_bias_29)
        self.freq_bias_30 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[29])
        freq_bias_all.append(self.freq_bias_30)
        self.freq_bias_31 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[30])
        freq_bias_all.append(self.freq_bias_31)
        self.freq_bias_32 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[31])
        freq_bias_all.append(self.freq_bias_32)
        self.freq_bias_33 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[32])
        freq_bias_all.append(self.freq_bias_33)
        self.freq_bias_34 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[33])
        freq_bias_all.append(self.freq_bias_34)
        self.freq_bias_35 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[34])
        freq_bias_all.append(self.freq_bias_35)
        self.freq_bias_36 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[35])
        freq_bias_all.append(self.freq_bias_36)
        self.freq_bias_37 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[36])
        freq_bias_all.append(self.freq_bias_37)
        self.freq_bias_38 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[37])
        freq_bias_all.append(self.freq_bias_38)
        self.freq_bias_39 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[38])
        freq_bias_all.append(self.freq_bias_39)
        self.freq_bias_40 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[39])
        freq_bias_all.append(self.freq_bias_40)
        self.freq_bias_41 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[40])
        freq_bias_all.append(self.freq_bias_41)
        self.freq_bias_42 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[41])
        freq_bias_all.append(self.freq_bias_42)
        self.freq_bias_43 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[42])
        freq_bias_all.append(self.freq_bias_43)
        self.freq_bias_44 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[43])
        freq_bias_all.append(self.freq_bias_44)
        self.freq_bias_45 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[44])
        freq_bias_all.append(self.freq_bias_45)
        self.freq_bias_46 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[45])
        freq_bias_all.append(self.freq_bias_46)
        self.freq_bias_47 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[46])
        freq_bias_all.append(self.freq_bias_47)
        self.freq_bias_48 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[47])
        freq_bias_all.append(self.freq_bias_48)
        self.freq_bias_49 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[48])
        freq_bias_all.append(self.freq_bias_49)
        self.freq_bias_50 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[49])
        freq_bias_all.append(self.freq_bias_50)



        return freq_bias_all





from .utils_motifs import encode_box_info
@registry.ROI_RELATION_PREDICTOR.register("MotifsLike_MBC")
class MotifsLike_MBC(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLike_MBC, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False




        # get model configs

        # self.rel_compress_all,self.ctx_compress_all = self.generate_muti_networks3()
        self.rel_compress_all = self.generate_muti_networks()
        #self.CE_loss = nn.CrossEntropyLoss()
        # self.CE_loss = SmoothBCEwLogits()
        if(config.ATA):
             self.CE_loss = generate_50_rel_loss_ata()
             # self.CE_loss = nn.CrossEntropyLoss()
        else:
             self.CE_loss = generate_50_rel_loss()
            # self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics)
        # if self.Knowledge_Transfer_Mode != 'None':
        #     self.NLL_Loss = nn.NLLLoss()
        #     self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
        #     self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()
        '''
        torch.int64
        torch.float16
        '''


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []

        # pos_embed = self.pos_embed(encode_box_info(proposals))
        # pair_pos_embeds = []
        #
        # times = 0
        # lastlen = 0
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))


        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        #
        # pair_pos = cat(pair_pos_embeds,dim = 0)

        ctx_gate = self.post_cat(prod_rep)




        if self.use_vision:  #true
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features





        '''begin to change'''
        add_losses = {}
        cls_num_list = [118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507,
                        3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580,
                        512, 511,
                        493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]

        if self.training:


            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj



            rel_labels = cat(rel_labels, dim=0)




            device = rel_labels.device
            mask_zero = torch.zeros_like(rel_labels)
            mask_one = torch.ones_like(rel_labels)



            loss_tmp = {}
            match_score = {}




            kl_first_reldist = None
            counter = 0




            for i in range(50):
                # rel_compress_test = self.rel_compress_all[i]
                # ctx_compress_test = self.ctx_compress_all[i]
                self.rel_compress_all[i].weight.requires_grad = True
                #.ctx_compress_all.weight.requires_grad = True
               # rel_dists = self.rel_compress_all[i](visual_rep) + self.ctx_compress_all[i](prod_rep)
                #rel_dists = self.rel_compress_all[i](visual_rep)+self.ctx_compress_all(whole_feature_gate)
                # rel_dists = self.rel_compress_all[i](visual_rep)
                tar_rel_label_group = torch.where(rel_labels==(i+1),mask_one,mask_zero)
                sum_rel = tar_rel_label_group.sum()

                if(sum_rel == 0):
                    add_losses['%d_CE_loss' % (i + 1)] = torch.tensor([0]).float().to(device)
                    self.rel_compress_all[i].weight.requires_grad = False
                    #self.ctx_compress_all.weight.requires_grad = False
                else:

                    rel_dists = self.rel_compress_all[i](visual_rep)

                    if self.use_bias:
                        rel_bias_now = self.freq_bias_all[i]
                        rel_dists = rel_dists + rel_bias_now.index_with_labels(pair_pred.long())



                    add_losses['%d_CE_loss' % (i + 1)] = self.CE_loss[i](rel_dists,tar_rel_label_group)

                    # add_losses['%d_CE_loss' % (i + 1)] =self.CE_loss(rel_dists,tar_rel_label_group)







            return None, None, add_losses

        else:
            obj_dists = obj_dists.split(num_objs, dim=0)

            rel_dists_sum = None

            lastbias_softmax = None
            for i in range(50):
               # rel_dists = self.rel_compress_all[i](visual_rep) + self.ctx_compress_all(whole_feature_gate)
                rel_dists = self.rel_compress_all[i](visual_rep)
                if self.use_bias:
                    rel_bias_test = self.freq_bias_all[i]
                    rel_dists = rel_dists+rel_bias_test.index_with_labels(pair_pred.long())
                    # rel_dists = self.pos_net(pair_pos)



                rel_scores_tmp = F.softmax(rel_dists, -1)


                if (rel_dists_sum is None):
                    rel_dists_sum =rel_scores_tmp
                else:

                    rel_dists_sum = torch.cat((rel_dists_sum,rel_scores_tmp[:,1:]),1)

            rel_dists_result = rel_dists_sum.split(num_rels, dim=0)

            return obj_dists, rel_dists_result, add_losses




    def getMatchScore(self,score,tar):
      match = score*tar
      tmp = match.sum()
      length = tar.sum()
      res = torch.true_divide(tmp,length)
      return res

    def calulateWeight(self,x):
        if(x == 0):
            return 60
        else:
            return -1 * (1 - x) * (1 - x) * np.math.log(x)

    def generate_muti_networks3(self):
        classifer_all = []

        self.rel_classifer_1 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_1, xavier=True)
        classifer_all.append(self.rel_classifer_1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_2, xavier=True)
        classifer_all.append(self.rel_classifer_2)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_3, xavier=True)
        classifer_all.append(self.rel_classifer_3)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_4, xavier=True)
        classifer_all.append(self.rel_classifer_4)
        self.rel_classifer_5 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_5, xavier=True)
        classifer_all.append(self.rel_classifer_5)
        self.rel_classifer_6 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_6, xavier=True)
        classifer_all.append(self.rel_classifer_6)
        self.rel_classifer_7 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_7, xavier=True)
        classifer_all.append(self.rel_classifer_7)
        self.rel_classifer_8 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_8, xavier=True)
        classifer_all.append(self.rel_classifer_8)
        self.rel_classifer_9 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_9, xavier=True)
        classifer_all.append(self.rel_classifer_9)
        self.rel_classifer_10 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_10, xavier=True)
        classifer_all.append(self.rel_classifer_10)
        self.rel_classifer_11 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_11, xavier=True)
        classifer_all.append(self.rel_classifer_11)
        self.rel_classifer_12 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_12, xavier=True)
        classifer_all.append(self.rel_classifer_12)
        self.rel_classifer_13 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_13, xavier=True)
        classifer_all.append(self.rel_classifer_13)
        self.rel_classifer_14 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_14, xavier=True)
        classifer_all.append(self.rel_classifer_14)
        self.rel_classifer_15 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_15, xavier=True)
        classifer_all.append(self.rel_classifer_15)
        self.rel_classifer_16 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_16, xavier=True)
        classifer_all.append(self.rel_classifer_16)
        self.rel_classifer_17 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_17, xavier=True)
        classifer_all.append(self.rel_classifer_17)
        self.rel_classifer_18 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_18, xavier=True)
        classifer_all.append(self.rel_classifer_18)
        self.rel_classifer_19 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_19, xavier=True)
        classifer_all.append(self.rel_classifer_19)
        self.rel_classifer_20 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_20, xavier=True)
        classifer_all.append(self.rel_classifer_20)
        self.rel_classifer_21 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_21, xavier=True)
        classifer_all.append(self.rel_classifer_21)
        self.rel_classifer_22 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_22, xavier=True)
        classifer_all.append(self.rel_classifer_22)
        self.rel_classifer_23 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_23, xavier=True)
        classifer_all.append(self.rel_classifer_23)
        self.rel_classifer_24 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_24, xavier=True)
        classifer_all.append(self.rel_classifer_24)
        self.rel_classifer_25 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_25, xavier=True)
        classifer_all.append(self.rel_classifer_25)
        self.rel_classifer_26 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_26, xavier=True)
        classifer_all.append(self.rel_classifer_26)
        self.rel_classifer_27 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_27, xavier=True)
        classifer_all.append(self.rel_classifer_27)
        self.rel_classifer_28 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_28, xavier=True)
        classifer_all.append(self.rel_classifer_28)
        self.rel_classifer_29 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_29, xavier=True)
        classifer_all.append(self.rel_classifer_29)
        self.rel_classifer_30 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_30, xavier=True)
        classifer_all.append(self.rel_classifer_30)
        self.rel_classifer_31 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_31, xavier=True)
        classifer_all.append(self.rel_classifer_31)
        self.rel_classifer_32 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_32, xavier=True)
        classifer_all.append(self.rel_classifer_32)
        self.rel_classifer_33 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_33, xavier=True)
        classifer_all.append(self.rel_classifer_33)
        self.rel_classifer_34 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_34, xavier=True)
        classifer_all.append(self.rel_classifer_34)
        self.rel_classifer_35 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_35, xavier=True)
        classifer_all.append(self.rel_classifer_35)
        self.rel_classifer_36 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_36, xavier=True)
        classifer_all.append(self.rel_classifer_36)
        self.rel_classifer_37 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_37, xavier=True)
        classifer_all.append(self.rel_classifer_37)
        self.rel_classifer_38 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_38, xavier=True)
        classifer_all.append(self.rel_classifer_38)
        self.rel_classifer_39 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_39, xavier=True)
        classifer_all.append(self.rel_classifer_39)
        self.rel_classifer_40 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_40, xavier=True)
        classifer_all.append(self.rel_classifer_40)
        self.rel_classifer_41 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_41, xavier=True)
        classifer_all.append(self.rel_classifer_41)
        self.rel_classifer_42 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_42, xavier=True)
        classifer_all.append(self.rel_classifer_42)
        self.rel_classifer_43 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_43, xavier=True)
        classifer_all.append(self.rel_classifer_43)
        self.rel_classifer_44 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_44, xavier=True)
        classifer_all.append(self.rel_classifer_44)
        self.rel_classifer_45 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_45, xavier=True)
        classifer_all.append(self.rel_classifer_45)
        self.rel_classifer_46 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_46, xavier=True)
        classifer_all.append(self.rel_classifer_46)
        self.rel_classifer_47 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_47, xavier=True)
        classifer_all.append(self.rel_classifer_47)
        self.rel_classifer_48 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_48, xavier=True)
        classifer_all.append(self.rel_classifer_48)
        self.rel_classifer_49 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_49, xavier=True)
        classifer_all.append(self.rel_classifer_49)
        self.rel_classifer_50 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_50, xavier=True)
        classifer_all.append(self.rel_classifer_50)

        self.rel_compress_1 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_1, xavier=True)


        return classifer_all,self.rel_compress_1
    def generate_muti_networks2(self):

        classifer_all = []
        compress_all = []

        self.rel_classifer_1 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_1, xavier=True)
        classifer_all.append(self.rel_classifer_1)
        self.rel_compress_1 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_1, xavier=True)
        compress_all.append(self.rel_compress_1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_2, xavier=True)
        classifer_all.append(self.rel_classifer_2)
        self.rel_compress_2 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_2, xavier=True)
        compress_all.append(self.rel_compress_2)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_3, xavier=True)
        classifer_all.append(self.rel_classifer_3)
        self.rel_compress_3 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_3, xavier=True)
        compress_all.append(self.rel_compress_3)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_4, xavier=True)
        classifer_all.append(self.rel_classifer_4)
        self.rel_compress_4 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_4, xavier=True)
        compress_all.append(self.rel_compress_4)
        self.rel_classifer_5 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_5, xavier=True)
        classifer_all.append(self.rel_classifer_5)
        self.rel_compress_5 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_5, xavier=True)
        compress_all.append(self.rel_compress_5)
        self.rel_classifer_6 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_6, xavier=True)
        classifer_all.append(self.rel_classifer_6)
        self.rel_compress_6 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_6, xavier=True)
        compress_all.append(self.rel_compress_6)
        self.rel_classifer_7 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_7, xavier=True)
        classifer_all.append(self.rel_classifer_7)
        self.rel_compress_7 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_7, xavier=True)
        compress_all.append(self.rel_compress_7)
        self.rel_classifer_8 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_8, xavier=True)
        classifer_all.append(self.rel_classifer_8)
        self.rel_compress_8 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_8, xavier=True)
        compress_all.append(self.rel_compress_8)
        self.rel_classifer_9 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_9, xavier=True)
        classifer_all.append(self.rel_classifer_9)
        self.rel_compress_9 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_9, xavier=True)
        compress_all.append(self.rel_compress_9)
        self.rel_classifer_10 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_10, xavier=True)
        classifer_all.append(self.rel_classifer_10)
        self.rel_compress_10 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_10, xavier=True)
        compress_all.append(self.rel_compress_10)
        self.rel_classifer_11 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_11, xavier=True)
        classifer_all.append(self.rel_classifer_11)
        self.rel_compress_11 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_11, xavier=True)
        compress_all.append(self.rel_compress_11)
        self.rel_classifer_12 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_12, xavier=True)
        classifer_all.append(self.rel_classifer_12)
        self.rel_compress_12 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_12, xavier=True)
        compress_all.append(self.rel_compress_12)
        self.rel_classifer_13 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_13, xavier=True)
        classifer_all.append(self.rel_classifer_13)
        self.rel_compress_13 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_13, xavier=True)
        compress_all.append(self.rel_compress_13)
        self.rel_classifer_14 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_14, xavier=True)
        classifer_all.append(self.rel_classifer_14)
        self.rel_compress_14 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_14, xavier=True)
        compress_all.append(self.rel_compress_14)
        self.rel_classifer_15 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_15, xavier=True)
        classifer_all.append(self.rel_classifer_15)
        self.rel_compress_15 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_15, xavier=True)
        compress_all.append(self.rel_compress_15)
        self.rel_classifer_16 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_16, xavier=True)
        classifer_all.append(self.rel_classifer_16)
        self.rel_compress_16 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_16, xavier=True)
        compress_all.append(self.rel_compress_16)
        self.rel_classifer_17 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_17, xavier=True)
        classifer_all.append(self.rel_classifer_17)
        self.rel_compress_17 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_17, xavier=True)
        compress_all.append(self.rel_compress_17)
        self.rel_classifer_18 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_18, xavier=True)
        classifer_all.append(self.rel_classifer_18)
        self.rel_compress_18 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_18, xavier=True)
        compress_all.append(self.rel_compress_18)
        self.rel_classifer_19 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_19, xavier=True)
        classifer_all.append(self.rel_classifer_19)
        self.rel_compress_19 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_19, xavier=True)
        compress_all.append(self.rel_compress_19)
        self.rel_classifer_20 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_20, xavier=True)
        classifer_all.append(self.rel_classifer_20)
        self.rel_compress_20 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_20, xavier=True)
        compress_all.append(self.rel_compress_20)
        self.rel_classifer_21 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_21, xavier=True)
        classifer_all.append(self.rel_classifer_21)
        self.rel_compress_21 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_21, xavier=True)
        compress_all.append(self.rel_compress_21)
        self.rel_classifer_22 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_22, xavier=True)
        classifer_all.append(self.rel_classifer_22)
        self.rel_compress_22 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_22, xavier=True)
        compress_all.append(self.rel_compress_22)
        self.rel_classifer_23 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_23, xavier=True)
        classifer_all.append(self.rel_classifer_23)
        self.rel_compress_23 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_23, xavier=True)
        compress_all.append(self.rel_compress_23)
        self.rel_classifer_24 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_24, xavier=True)
        classifer_all.append(self.rel_classifer_24)
        self.rel_compress_24 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_24, xavier=True)
        compress_all.append(self.rel_compress_24)
        self.rel_classifer_25 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_25, xavier=True)
        classifer_all.append(self.rel_classifer_25)
        self.rel_compress_25 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_25, xavier=True)
        compress_all.append(self.rel_compress_25)
        self.rel_classifer_26 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_26, xavier=True)
        classifer_all.append(self.rel_classifer_26)
        self.rel_compress_26 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_26, xavier=True)
        compress_all.append(self.rel_compress_26)
        self.rel_classifer_27 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_27, xavier=True)
        classifer_all.append(self.rel_classifer_27)
        self.rel_compress_27 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_27, xavier=True)
        compress_all.append(self.rel_compress_27)
        self.rel_classifer_28 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_28, xavier=True)
        classifer_all.append(self.rel_classifer_28)
        self.rel_compress_28 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_28, xavier=True)
        compress_all.append(self.rel_compress_28)
        self.rel_classifer_29 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_29, xavier=True)
        classifer_all.append(self.rel_classifer_29)
        self.rel_compress_29 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_29, xavier=True)
        compress_all.append(self.rel_compress_29)
        self.rel_classifer_30 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_30, xavier=True)
        classifer_all.append(self.rel_classifer_30)
        self.rel_compress_30 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_30, xavier=True)
        compress_all.append(self.rel_compress_30)
        self.rel_classifer_31 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_31, xavier=True)
        classifer_all.append(self.rel_classifer_31)
        self.rel_compress_31 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_31, xavier=True)
        compress_all.append(self.rel_compress_31)
        self.rel_classifer_32 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_32, xavier=True)
        classifer_all.append(self.rel_classifer_32)
        self.rel_compress_32 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_32, xavier=True)
        compress_all.append(self.rel_compress_32)
        self.rel_classifer_33 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_33, xavier=True)
        classifer_all.append(self.rel_classifer_33)
        self.rel_compress_33 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_33, xavier=True)
        compress_all.append(self.rel_compress_33)
        self.rel_classifer_34 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_34, xavier=True)
        classifer_all.append(self.rel_classifer_34)
        self.rel_compress_34 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_34, xavier=True)
        compress_all.append(self.rel_compress_34)
        self.rel_classifer_35 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_35, xavier=True)
        classifer_all.append(self.rel_classifer_35)
        self.rel_compress_35 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_35, xavier=True)
        compress_all.append(self.rel_compress_35)
        self.rel_classifer_36 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_36, xavier=True)
        classifer_all.append(self.rel_classifer_36)
        self.rel_compress_36 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_36, xavier=True)
        compress_all.append(self.rel_compress_36)
        self.rel_classifer_37 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_37, xavier=True)
        classifer_all.append(self.rel_classifer_37)
        self.rel_compress_37 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_37, xavier=True)
        compress_all.append(self.rel_compress_37)
        self.rel_classifer_38 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_38, xavier=True)
        classifer_all.append(self.rel_classifer_38)
        self.rel_compress_38 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_38, xavier=True)
        compress_all.append(self.rel_compress_38)
        self.rel_classifer_39 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_39, xavier=True)
        classifer_all.append(self.rel_classifer_39)
        self.rel_compress_39 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_39, xavier=True)
        compress_all.append(self.rel_compress_39)
        self.rel_classifer_40 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_40, xavier=True)
        classifer_all.append(self.rel_classifer_40)
        self.rel_compress_40 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_40, xavier=True)
        compress_all.append(self.rel_compress_40)
        self.rel_classifer_41 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_41, xavier=True)
        classifer_all.append(self.rel_classifer_41)
        self.rel_compress_41 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_41, xavier=True)
        compress_all.append(self.rel_compress_41)
        self.rel_classifer_42 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_42, xavier=True)
        classifer_all.append(self.rel_classifer_42)
        self.rel_compress_42 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_42, xavier=True)
        compress_all.append(self.rel_compress_42)
        self.rel_classifer_43 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_43, xavier=True)
        classifer_all.append(self.rel_classifer_43)
        self.rel_compress_43 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_43, xavier=True)
        compress_all.append(self.rel_compress_43)
        self.rel_classifer_44 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_44, xavier=True)
        classifer_all.append(self.rel_classifer_44)
        self.rel_compress_44 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_44, xavier=True)
        compress_all.append(self.rel_compress_44)
        self.rel_classifer_45 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_45, xavier=True)
        classifer_all.append(self.rel_classifer_45)
        self.rel_compress_45 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_45, xavier=True)
        compress_all.append(self.rel_compress_45)
        self.rel_classifer_46 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_46, xavier=True)
        classifer_all.append(self.rel_classifer_46)
        self.rel_compress_46 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_46, xavier=True)
        compress_all.append(self.rel_compress_46)
        self.rel_classifer_47 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_47, xavier=True)
        classifer_all.append(self.rel_classifer_47)
        self.rel_compress_47 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_47, xavier=True)
        compress_all.append(self.rel_compress_47)
        self.rel_classifer_48 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_48, xavier=True)
        classifer_all.append(self.rel_classifer_48)
        self.rel_compress_48 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_48, xavier=True)
        compress_all.append(self.rel_compress_48)
        self.rel_classifer_49 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_49, xavier=True)
        classifer_all.append(self.rel_classifer_49)
        self.rel_compress_49 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_49, xavier=True)
        compress_all.append(self.rel_compress_49)
        self.rel_classifer_50 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_50, xavier=True)
        classifer_all.append(self.rel_classifer_50)
        self.rel_compress_50 = nn.Linear(4096, 2)
        layer_init(self.rel_compress_50, xavier=True)
        compress_all.append(self.rel_compress_50)


        return classifer_all,compress_all

    def generate_muti_networks(self):
        classifer_all = []

        self.rel_classifer_1 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_1, xavier=True)
        classifer_all.append(self.rel_classifer_1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_2, xavier=True)
        classifer_all.append(self.rel_classifer_2)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_3, xavier=True)
        classifer_all.append(self.rel_classifer_3)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_4, xavier=True)
        classifer_all.append(self.rel_classifer_4)
        self.rel_classifer_5 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_5, xavier=True)
        classifer_all.append(self.rel_classifer_5)
        self.rel_classifer_6 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_6, xavier=True)
        classifer_all.append(self.rel_classifer_6)
        self.rel_classifer_7 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_7, xavier=True)
        classifer_all.append(self.rel_classifer_7)
        self.rel_classifer_8 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_8, xavier=True)
        classifer_all.append(self.rel_classifer_8)
        self.rel_classifer_9 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_9, xavier=True)
        classifer_all.append(self.rel_classifer_9)
        self.rel_classifer_10 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_10, xavier=True)
        classifer_all.append(self.rel_classifer_10)
        self.rel_classifer_11 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_11, xavier=True)
        classifer_all.append(self.rel_classifer_11)
        self.rel_classifer_12 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_12, xavier=True)
        classifer_all.append(self.rel_classifer_12)
        self.rel_classifer_13 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_13, xavier=True)
        classifer_all.append(self.rel_classifer_13)
        self.rel_classifer_14 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_14, xavier=True)
        classifer_all.append(self.rel_classifer_14)
        self.rel_classifer_15 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_15, xavier=True)
        classifer_all.append(self.rel_classifer_15)
        self.rel_classifer_16 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_16, xavier=True)
        classifer_all.append(self.rel_classifer_16)
        self.rel_classifer_17 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_17, xavier=True)
        classifer_all.append(self.rel_classifer_17)
        self.rel_classifer_18 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_18, xavier=True)
        classifer_all.append(self.rel_classifer_18)
        self.rel_classifer_19 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_19, xavier=True)
        classifer_all.append(self.rel_classifer_19)
        self.rel_classifer_20 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_20, xavier=True)
        classifer_all.append(self.rel_classifer_20)
        self.rel_classifer_21 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_21, xavier=True)
        classifer_all.append(self.rel_classifer_21)
        self.rel_classifer_22 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_22, xavier=True)
        classifer_all.append(self.rel_classifer_22)
        self.rel_classifer_23 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_23, xavier=True)
        classifer_all.append(self.rel_classifer_23)
        self.rel_classifer_24 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_24, xavier=True)
        classifer_all.append(self.rel_classifer_24)
        self.rel_classifer_25 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_25, xavier=True)
        classifer_all.append(self.rel_classifer_25)
        self.rel_classifer_26 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_26, xavier=True)
        classifer_all.append(self.rel_classifer_26)
        self.rel_classifer_27 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_27, xavier=True)
        classifer_all.append(self.rel_classifer_27)
        self.rel_classifer_28 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_28, xavier=True)
        classifer_all.append(self.rel_classifer_28)
        self.rel_classifer_29 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_29, xavier=True)
        classifer_all.append(self.rel_classifer_29)
        self.rel_classifer_30 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_30, xavier=True)
        classifer_all.append(self.rel_classifer_30)
        self.rel_classifer_31 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_31, xavier=True)
        classifer_all.append(self.rel_classifer_31)
        self.rel_classifer_32 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_32, xavier=True)
        classifer_all.append(self.rel_classifer_32)
        self.rel_classifer_33 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_33, xavier=True)
        classifer_all.append(self.rel_classifer_33)
        self.rel_classifer_34 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_34, xavier=True)
        classifer_all.append(self.rel_classifer_34)
        self.rel_classifer_35 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_35, xavier=True)
        classifer_all.append(self.rel_classifer_35)
        self.rel_classifer_36 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_36, xavier=True)
        classifer_all.append(self.rel_classifer_36)
        self.rel_classifer_37 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_37, xavier=True)
        classifer_all.append(self.rel_classifer_37)
        self.rel_classifer_38 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_38, xavier=True)
        classifer_all.append(self.rel_classifer_38)
        self.rel_classifer_39 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_39, xavier=True)
        classifer_all.append(self.rel_classifer_39)
        self.rel_classifer_40 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_40, xavier=True)
        classifer_all.append(self.rel_classifer_40)
        self.rel_classifer_41 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_41, xavier=True)
        classifer_all.append(self.rel_classifer_41)
        self.rel_classifer_42 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_42, xavier=True)
        classifer_all.append(self.rel_classifer_42)
        self.rel_classifer_43 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_43, xavier=True)
        classifer_all.append(self.rel_classifer_43)
        self.rel_classifer_44 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_44, xavier=True)
        classifer_all.append(self.rel_classifer_44)
        self.rel_classifer_45 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_45, xavier=True)
        classifer_all.append(self.rel_classifer_45)
        self.rel_classifer_46 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_46, xavier=True)
        classifer_all.append(self.rel_classifer_46)
        self.rel_classifer_47 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_47, xavier=True)
        classifer_all.append(self.rel_classifer_47)
        self.rel_classifer_48 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_48, xavier=True)
        classifer_all.append(self.rel_classifer_48)
        self.rel_classifer_49 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_49, xavier=True)
        classifer_all.append(self.rel_classifer_49)
        self.rel_classifer_50 = nn.Linear(self.pooling_dim, 2)
        layer_init(self.rel_classifer_50, xavier=True)
        classifer_all.append(self.rel_classifer_50)

        return classifer_all

    def generate_multi_bias(self, config, statistics):
        freq_bias_all = []
        predicate_all_list = []

        for i in range(50):
            list_tmp = [0 for j in range(51)]
            list_tmp[i + 1] = i + 1
            predicate_all_list.append(list_tmp)

        self.freq_bias_1 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[0])
        freq_bias_all.append(self.freq_bias_1)
        self.freq_bias_2 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[1])
        freq_bias_all.append(self.freq_bias_2)
        self.freq_bias_3 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[2])
        freq_bias_all.append(self.freq_bias_3)
        self.freq_bias_4 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[3])
        freq_bias_all.append(self.freq_bias_4)
        self.freq_bias_5 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[4])
        freq_bias_all.append(self.freq_bias_5)
        self.freq_bias_6 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[5])
        freq_bias_all.append(self.freq_bias_6)
        self.freq_bias_7 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[6])
        freq_bias_all.append(self.freq_bias_7)
        self.freq_bias_8 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[7])
        freq_bias_all.append(self.freq_bias_8)
        self.freq_bias_9 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                               predicate_all_list=predicate_all_list[8])
        freq_bias_all.append(self.freq_bias_9)
        self.freq_bias_10 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[9])
        freq_bias_all.append(self.freq_bias_10)
        self.freq_bias_11 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[10])
        freq_bias_all.append(self.freq_bias_11)
        self.freq_bias_12 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[11])
        freq_bias_all.append(self.freq_bias_12)
        self.freq_bias_13 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[12])
        freq_bias_all.append(self.freq_bias_13)
        self.freq_bias_14 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[13])
        freq_bias_all.append(self.freq_bias_14)
        self.freq_bias_15 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[14])
        freq_bias_all.append(self.freq_bias_15)
        self.freq_bias_16 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[15])
        freq_bias_all.append(self.freq_bias_16)
        self.freq_bias_17 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[16])
        freq_bias_all.append(self.freq_bias_17)
        self.freq_bias_18 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[17])
        freq_bias_all.append(self.freq_bias_18)
        self.freq_bias_19 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[18])
        freq_bias_all.append(self.freq_bias_19)
        self.freq_bias_20 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[19])
        freq_bias_all.append(self.freq_bias_20)
        self.freq_bias_21 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[20])
        freq_bias_all.append(self.freq_bias_21)
        self.freq_bias_22 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[21])
        freq_bias_all.append(self.freq_bias_22)
        self.freq_bias_23 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[22])
        freq_bias_all.append(self.freq_bias_23)
        self.freq_bias_24 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[23])
        freq_bias_all.append(self.freq_bias_24)
        self.freq_bias_25 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[24])
        freq_bias_all.append(self.freq_bias_25)
        self.freq_bias_26 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[25])
        freq_bias_all.append(self.freq_bias_26)
        self.freq_bias_27 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[26])
        freq_bias_all.append(self.freq_bias_27)
        self.freq_bias_28 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[27])
        freq_bias_all.append(self.freq_bias_28)
        self.freq_bias_29 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[28])
        freq_bias_all.append(self.freq_bias_29)
        self.freq_bias_30 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[29])
        freq_bias_all.append(self.freq_bias_30)
        self.freq_bias_31 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[30])
        freq_bias_all.append(self.freq_bias_31)
        self.freq_bias_32 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[31])
        freq_bias_all.append(self.freq_bias_32)
        self.freq_bias_33 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[32])
        freq_bias_all.append(self.freq_bias_33)
        self.freq_bias_34 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[33])
        freq_bias_all.append(self.freq_bias_34)
        self.freq_bias_35 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[34])
        freq_bias_all.append(self.freq_bias_35)
        self.freq_bias_36 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[35])
        freq_bias_all.append(self.freq_bias_36)
        self.freq_bias_37 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[36])
        freq_bias_all.append(self.freq_bias_37)
        self.freq_bias_38 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[37])
        freq_bias_all.append(self.freq_bias_38)
        self.freq_bias_39 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[38])
        freq_bias_all.append(self.freq_bias_39)
        self.freq_bias_40 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[39])
        freq_bias_all.append(self.freq_bias_40)
        self.freq_bias_41 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[40])
        freq_bias_all.append(self.freq_bias_41)
        self.freq_bias_42 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[41])
        freq_bias_all.append(self.freq_bias_42)
        self.freq_bias_43 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[42])
        freq_bias_all.append(self.freq_bias_43)
        self.freq_bias_44 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[43])
        freq_bias_all.append(self.freq_bias_44)
        self.freq_bias_45 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[44])
        freq_bias_all.append(self.freq_bias_45)
        self.freq_bias_46 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[45])
        freq_bias_all.append(self.freq_bias_46)
        self.freq_bias_47 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[46])
        freq_bias_all.append(self.freq_bias_47)
        self.freq_bias_48 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[47])
        freq_bias_all.append(self.freq_bias_48)
        self.freq_bias_49 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[48])
        freq_bias_all.append(self.freq_bias_49)
        self.freq_bias_50 = FrequencyBias_Group(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                predicate_all_list=predicate_all_list[49])
        freq_bias_all.append(self.freq_bias_50)



        return freq_bias_all



@registry.ROI_RELATION_PREDICTOR.register("TransLike_GCL")
class TransLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_GCL, self).__init__()
        # load parameters
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.union_single_not_match:
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

            return None, None, add_losses
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

from .loss import make_roi_relation_loss_evaluator, create_CBLoss, choose_rel_loss, generate_50_rel_loss,generate_50_rel_loss_ata, \
    SmoothBCEwLogits




@registry.ROI_RELATION_PREDICTOR.register("MotifsLikePredictor")
class MotifsLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.criterion_rel_loss = choose_rel_loss()
        # self.criterion_rel_loss = nn.CrossEntropyLoss()
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:

                prod_rep = prod_rep * union_features



        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        add_losses = {}

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:  #No Run
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_rel_loss(rel_dists, rel_labels)

            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifsLike_GCL")
class MotifsLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifsLike_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'VTransE':
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            exit('wrong mode!')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()
        '''
        torch.int64
        torch.float16
        '''

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,features=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        '''begin to change'''
        add_losses = {}
        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy loss'''
                jdx = i
                rel_classier_now = self.rel_classifer_all[jdx]
                group_output_now = rel_classier_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            rel_classier_test = self.rel_classifer_all[-1]
            rel_dists = rel_classier_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all
#
@registry.ROI_RELATION_PREDICTOR.register("VCTree_GCL")
class VCTree_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTree_GCL, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_bias = config.GLOBAL_SETTING.USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_classifer_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()
        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)
        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        prod_rep = prod_rep * union_features

        add_losses = {}
        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy loss'''
                jdx = i
                rel_classier_now = self.rel_classifer_all[jdx]
                group_output_now = rel_classier_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_classier_bef = self.rel_classifer_all[jbef]
                        group_output_bef = rel_classier_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        # kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
            return None, None, add_losses
        else:
            rel_classier_test = self.rel_classifer_all[-1]
            rel_dists = rel_classier_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses


        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists

        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("TransLikePredictor")
class TransLikePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLikePredictor, self).__init__()
        self.config = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.VG_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        add_losses = {}

        if self.training:
            if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj
            rel_labels = cat(rel_labels, dim=0)
            add_losses['rel_loss'] = self.criterion_loss(rel_dists, rel_labels)
            return None, None, add_losses
        else:
            obj_dists = obj_dists.split(num_objs, dim=0)
            rel_dists = rel_dists.split(num_rels, dim=0)
            return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    import time
    result_str = '---'*20
    result_str += ('\n\nthe dataset we use is [ %s ]' % cfg.GLOBAL_SETTING.DATASET_CHOICE)
    if cfg.GLOBAL_SETTING.USE_BIAS:
        result_str += ('\nwe use [ bias ]!')
    else:
        result_str += ('\nwe do [ not ] use bias!')
    result_str += ('\nthe model we use is [ %s ]' % cfg.GLOBAL_SETTING.RELATION_PREDICTOR)
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ predcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
        result_str += ('\ntraining mode is [ sgcls ]')
    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == False:
        result_str += ('\ntraining mode is [ sgdet ]')
    else:
        exit('wrong training mode!')
    result_str += ('\nlearning rate is [ %.5f ]' % cfg.SOLVER.BASE_LR)
    result_str += ('\nthe knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    assert cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE in ['None', 'KL_logit_Neighbor', 'KL_logit_None',
                                               'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['TransLike_GCL', 'TransLikePredictor']:
        result_str += ('\nrel labels=0 is use [ %s ] to process' % cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE)
        assert cfg.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE in ['rand_insert', 'rand_choose', 'all_include']
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Self-Attention', 'Cross-Attention', 'Hybrid-Attention']
        result_str += ('\n-----Transformer layer is [ %d ] in obj and [ %d ] in rel' %
                       (cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER,
                        cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER))
        result_str += ('\n-----Transformer mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)
    if cfg.GLOBAL_SETTING.RELATION_PREDICTOR in ['MotifsLike_GCL', 'MotifsLikePredictor']:
        assert cfg.GLOBAL_SETTING.BASIC_ENCODER in ['Motifs', 'VTransE']
        result_str += ('\n-----Model mode is [ %s ]' % cfg.GLOBAL_SETTING.BASIC_ENCODER)

    num_of_group_element_list, predicate_stage_count = get_group_splits(cfg.GLOBAL_SETTING.DATASET_CHOICE, cfg.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE)
    max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
    incre_idx_list, max_elemnt_list, group_matrix, kd_matrix = get_current_predicate_idx(
        num_of_group_element_list, cfg.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_PENALTY, cfg.GLOBAL_SETTING.DATASET_CHOICE)
    result_str += ('\n   the number of elements in each group is {}'.format(incre_idx_list))
    result_str += ('\n   incremental stage list is {}'.format(num_of_group_element_list))
    result_str += ('\n   the length of each line in group is {}'.format(predicate_stage_count))
    result_str += ('\n   the max number of elements in each group is {}'.format(max_group_element_number_list))
    result_str += ('\n   the knowledge distillation strategy is [ %s ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE)
    result_str += ('\n   the penalty for whole distillation loss is [ %.2f ]' % cfg.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT)
    with open(os.path.join(cfg.OUTPUT_DIR, 'control_info.txt'), 'w') as outfile:
        outfile.write(result_str)
    result_str += '\n\n'
    result_str += '---'*20
    print(result_str)
    time.sleep(2)
    func = registry.ROI_RELATION_PREDICTOR[cfg.GLOBAL_SETTING.RELATION_PREDICTOR]
    return func(cfg, in_channels)
