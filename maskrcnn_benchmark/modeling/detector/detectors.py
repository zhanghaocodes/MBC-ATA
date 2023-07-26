# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_Transfer import GeneralizedRCNNTransfer

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)



_DETECTION_META_ARCHITECTURES_TRANSFER = {"GeneralizedRCNNTransfer": GeneralizedRCNNTransfer}

def build_detection_model_transfer(cfg,object_class_list,pre_class_list):
    meta_arch = _DETECTION_META_ARCHITECTURES_TRANSFER[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg,object_class_list,pre_class_list)