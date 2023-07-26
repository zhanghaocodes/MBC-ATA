# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import os
from maskrcnn_benchmark.modeling.utils import cat
import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from ..roi_heads.relation_head.sampling import make_roi_relation_samp_processor
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from ..roi_heads.relation_head.utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info


import pandas as pd


class GeneralizedRCNNTransfer(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg,object_list,pre_class_list):
        super(GeneralizedRCNNTransfer, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone.out_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        #计算roi feature
        self.box_feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone.out_channels)
        #word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.pre_class_list = pre_class_list
        self.num_obj_classes = len(object_list)
        self.obj_classes = object_list
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.sumTripleFeature,self.numTripleDict = self.initDict()
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)


        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])


    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        #下面是得到三元组的特征向量:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        with torch.no_grad():
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, features, targets)

            x, detections, loss_box = self.box(features, proposals, targets)
            proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(detections, targets)
            roi_features = self.box_feature_extractor(features, proposals)
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_embed = self.obj_embed1(obj_labels.long())

            pos_embed = self.pos_embed(encode_box_info(proposals))

            obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
            # 找到三元组的标签并计算向量保存进dict中   计算平均特征向量
            # p = 0
            # Lastlength = 0
            # for i in range(len(targets)):  # 处理每一张图片
            #     target = targets[i]
            #     object = target.get_field('labels')
            #     if (i > 0):
            #         p += Lastlength
            #     Lastlength = len(object)
            #     relations = target.get_field('relation')
            #
            #     for row in range(len(relations)):
            #         for col in range(len(relations)):
            #             if relations[row][col] != 0:
            #                 key = str(self.obj_classes[object[row].item()]) + "," + str(
            #                     self.pre_class_list[relations[row][col].item()]) + "," + str(
            #                     self.obj_classes[object[col].item()])
            #                 sub_obj_pre = cat((obj_pre_rep[row + p], obj_pre_rep[col + p]), -1).cpu()
            #                 self.sumTripleFeature[key] = self.sumTripleFeature.get(key) + sub_obj_pre
            #                 self.numTripleDict[key] = self.numTripleDict.get(key) + 1







        return  obj_pre_rep


    def initDict(self):
        dict = {}
        numDict = {}
        init_array = torch.zeros(8848)

        dict_tmp = torch.load("TripleNumber0707.pth")
        keys = list(dict_tmp.keys())
        for i in range(len(keys)):
            numDict[keys[i]] = 0
            dict[keys[i]] = init_array

        return dict, numDict



    def box(self,features,proposals,targets):
        proposals = [target.copy_with_fields(["labels"]) for target in targets]
        x = self.feature_extractor(features, proposals)
        return x, proposals, {}