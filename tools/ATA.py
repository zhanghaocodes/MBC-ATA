# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys

#sys.path.append(r"/home/share/zhanghao/codes/MBC-ATA")
from IPython.core.display import display
from apex.amp import amp

from maskrcnn_benchmark.data.datasets.visual_genome import box_filter
from maskrcnn_benchmark.structures.image_list import to_image_list


from maskrcnn_benchmark.modeling.detector.detectors import build_detection_model_transfer
from maskrcnn_benchmark.modeling.roi_heads.relation_head.sampling import make_roi_relation_samp_processor


from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


from maskrcnn_benchmark.modeling.utils import cat
import pandas as pd
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import numpy as np
import random
def export_excel_direct(dic_data,path):
    # 将字典列表转换为DataFrame
    key = list(dic_data.keys())
    value = list(dic_data.values())
    pf = pd.DataFrame()


    pf["name1"] = key
    pf["name2"]=value

    file_path = pd.ExcelWriter(path)

    pf.fillna(' ', inplace=True)
    pf.to_excel(file_path, encoding='utf-8', index=False)
    file_path.save()

def image_box_overlap(boxes, query_boxes, criterion=-1):
    """
    计算图像box的iou
    Args:
        boxes:一个part中的全部gt，以第一个part为例(642,4)
        query_boxes：一个part中的全部dt，以第一个part为例(233,4)
    """
    N = boxes.shape[0] # gt_box的总数
    K = query_boxes.shape[0] # det_box的总数
    # 初始化overlap矩阵（x1，y1，x2，y2）左上右下两个点，query_boxes类似如下矩阵
    # 代码中每个part是进行批量的处理
    ''''bbox': 
        array([[548.  , 171.33, 572.4 , 194.42],
               [505.25, 168.37, 575.44, 209.18],
               [ 49.7 , 185.65, 227.42, 246.96],
               [328.67, 170.65, 397.24, 204.16],
               [603.36, 169.62, 631.06, 186.56],
               [578.97, 168.88, 603.78, 187.56]], dtype=float32),'''
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    # overlaps 为二维数组（642,233）
    # 两层for循环逐个box计算iou，因为有jit加速，所以成for循环的形式
    for k in range(K):
        # 计算第k个dt box的面积（box是左上角和右下角的形式[x1,y1,x2,y2]）
        # (x2-x1)*(y2-y1)
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N): # 遍历gt boxes
            # 重叠部分的宽度 = 两个图像右边缘的较小值 - 两个图像左边缘的较大值
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0: # 如果宽度方向有重叠，再计算高度
                # 重叠部分的高度 = 两个图像上边缘的较小值 - 两个图像下边缘的较大值
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1: # 默认执行criterion = -1
                        # 求两个box的并
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    # 计算iou = 交集/并集
                    overlaps[n, k] = iw * ih / ua
    return overlaps
def scalebox (box,img_info):
    scale_num  = 0.7 +0.6*random.random()


    x1,y1,x2,y2 = box
    x_c = (x2 + x1)/2
    y_c = (y1+y2)/2
    width = x2-x1
    height = y2-y1

    big_width = scale_num*width
    big_height = scale_num*height

    new_x1 = x_c-0.5*big_width
    new_y1 = y_c-0.5*big_height
    new_x2 = x_c + 0.5 * big_width
    new_y2 = y_c + 0.5 * big_height

    if(new_x1>0 and new_y1>0 and new_x2>0 and new_y2>0 and new_x1<img_info['width'] and new_y1 <img_info['height'] and new_x2<img_info['width'] and new_y2 <img_info['height']):
        box = np.asarray([new_x1, new_y1, new_x2, new_y2])
        return box
    else:
        scale_num = 0.8 + 0.2 * random.random()
        big_width = scale_num * width
        big_height = scale_num * height

        new_x1 = x_c - 0.5 * big_width
        new_y1 = y_c - 0.5 * big_height
        new_x2 = x_c + 0.5 * big_width
        new_y2 = y_c + 0.5 * big_height

        box = np.asarray([new_x1, new_y1, new_x2, new_y2])
        return box
def translationbox(box, img_info):
    offset_x =20*random.random()-10
    offset_y = 20*random.random()-10

    x1, y1, x2, y2 = box
    x_c = (x2 + x1) / 2
    y_c = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    x_c_new =x_c+offset_x
    y_c_new = y_c+offset_y

    new_x1 = x_c_new - 0.5 * width
    new_y1 = y_c_new - 0.5 * height
    new_x2 = x_c_new + 0.5 * width
    new_y2 = y_c_new + 0.5 * height

    if (new_x1 >= 0 and new_y1 >= 0 and new_x2 >= 0 and new_y2 >= 0 and new_x1 < img_info['width'] and new_y1 < img_info[
        'height'] and new_x2 < img_info['width'] and new_y2 < img_info['height']):
        box = np.asarray([new_x1, new_y1, new_x2, new_y2])
        return box
    else:
        box = np.asarray([new_x1, new_y1, new_x2, new_y2])
        for i in range(0,4):
            if(i<0):
                box[i] = 0
            if((i==0 or i==2)and i>img_info['width']):
                box[i] = img_info['width']-1
            if ((i == 1 or i == 3) and i > img_info['height']):
                box[i] = img_info['height'] -1




        return box
def createTranslationList (box,iou,image_info):
    BOX_SCALE = 1024
    res = []
    image_width = image_info['width']
    image_height = image_info['height']
    w = box[2]-box[0]
    h = box[3]-box[1]

    offset_y = h*(1-iou)/(1+iou)
    offset_x = w*(1-iou)/(1+iou)

    t_0 = np.asarray([box[0],box[1]+offset_y,box[2],box[3]+offset_y])
    t_0_orininal = t_0/BOX_SCALE*max(image_width,image_height)
    if(t_0_orininal[3]>image_height):
        t_0[3] = box[3]
    t_1 = np.asarray([box[0], box[1]-offset_y, box[2], box[3]-offset_y])
    t_1_orininal = t_1 / BOX_SCALE * max(image_width, image_height)
    if(t_1_orininal[1]< 0):
        t_1[1] = box[1]
    t_2 = np.asarray([box[0]+offset_x, box[1], box[2]+offset_x, box[3]])
    t_2_orininal = t_2 / BOX_SCALE * max(image_width, image_height)
    if(t_2_orininal[2]>image_width):
        t_2[2] = box[2]
    t_3 = np.asarray([box[0]-offset_x, box[1], box[2]-offset_x, box[3]])
    t_3_orininal = t_3 / BOX_SCALE * max(image_width, image_height)
    if (t_3_orininal[0] <0):
        t_3[0] = box[0]
    res.append(t_0)
    res.append(t_1)
    res.append(t_2)
    res.append(t_3)


    return res
def createScaleList (box,iou,image_info):
    BOX_SCALE = 1024
    res = []
    image_width = image_info['width']
    image_height = image_info['height']
    w = box[2]-box[0]
    h = box[3]-box[1]


    offset_x_small = (1-pow(iou,0.5))*w*0.5
    offset_y_small = (1-pow(iou,0.5))*h*0.5

    t_0 = np.asarray([box[0]+offset_x_small,box[1]+offset_y_small,box[2]-offset_x_small,box[3]-offset_y_small])

    offset_x_large = (pow(1/iou, 0.5)-1) * w * 0.5
    offset_y_large = (pow(1/iou, 0.5)-1) * h * 0.5
    t_1 = np.asarray([box[0] - offset_x_large, box[1] - offset_y_large, box[2] + offset_x_large, box[3] + offset_y_large])
    t_1_orininal = t_1 / BOX_SCALE * max(image_width, image_height)
    if(t_1_orininal[0]<0):
        t_1[0] = box[0]
    if(t_1_orininal[1]<0):
        t_1[1] = box[1]

    if(t_1_orininal[2]>image_width):
        t_1[2] = box[2]
    if (t_1_orininal[3] > image_height):
        t_1[3] = box[3]


    res.append(t_0)
    res.append(t_1)


    return res
def copyBox(box):
    res = []
    res.append(box)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/home/share/zhanghao/codes/MBC-ATA/configs/ATA.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1



    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(),filename="GetClassNumLog.txt")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())



    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)



    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        dataset_names = cfg.DATASETS.VG_TEST
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        dataset_names = cfg.DATASETS.GQA_200_TEST
    else:
        dataset_names = None
        exit('wrong Dataset name!')

    output_folders = [None] * len(dataset_names)



    data_loader = make_data_loader(cfg, mode="train", is_distributed=distributed)
    dataset_vg = data_loader.dataset  #train
    #dataset_vg = data_loader[0].dataset
    object_class_list = dataset_vg.ind_to_classes
    predicate_class_list = dataset_vg.ind_to_predicates

    dict_pred_num = {}

    lenth = len(dataset_vg.gt_boxes)
    for id in tqdm(range(0, lenth)):
        gt_boxes = list(dataset_vg.gt_boxes[id])
        gt_classes = list(dataset_vg.gt_classes[id])
        img_info = dataset_vg.img_info[id]
        relation = list(dataset_vg.relationships[id])
        map = {}
        for rel in range(len(relation)): #遍历每一个关系三元组
            pred = relation[rel][2]
            if(pred<25):
                continue
            else:
                sub = relation[rel][0]
                obj = relation[rel][1]
                if(sub>len(gt_boxes) or obj>len(gt_boxes)):
                    continue

                sub_box = gt_boxes[sub]
                obj_box = gt_boxes[obj]

                iou = 0.8
                new_sub_box_list = createTranslationList(sub_box,iou,img_info)
                new_sub_box_list.extend(createScaleList(sub_box,iou,img_info))

                k_num_box = 1
                new_sub_box = random.sample(new_sub_box_list,k_num_box)
                # new_sub_box = [sub_box]


                new_obj_box_list = createTranslationList(obj_box,iou,img_info)
                new_obj_box_list.extend(createScaleList(obj_box,iou,img_info))
                new_obj_box = random.sample(new_obj_box_list, k_num_box)
                # new_obj_box = [obj_box]

                if(not map):
                    map[sub] =[]
                    map[obj] = []
                    for i in range(k_num_box):
                        map[sub].append(len(gt_boxes)+i)
                        map[obj].append(len(gt_boxes) +k_num_box+ i)
                    gt_boxes.extend(new_sub_box)
                    gt_boxes.extend(new_obj_box)
                    gt_classes.extend([gt_classes[sub] for i in range(k_num_box)])
                    gt_classes.extend([gt_classes[obj] for i in range(k_num_box)])


                    for i in range(k_num_box):
                        relation.append(np.asarray([map[sub][i], map[obj][i], relation[rel][2]]))
                        relation.append(np.asarray([sub, map[obj][i], relation[rel][2]]))
                        relation.append(np.asarray([map[sub][i], obj, relation[rel][2]]))

                else:
                    map_keys = map.keys()
                    if(map_keys.__contains__(sub) and map_keys.__contains__(obj)):
                        for i in range(k_num_box):
                            relation.append(np.asarray([map[sub][i], map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([sub, map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([map[sub][i], obj, relation[rel][2]]))

                    elif (map_keys.__contains__(sub) and not map_keys.__contains__(obj)):
                        map[obj] = []
                        for i in range(k_num_box):
                             map[obj].append(len(gt_boxes) + i)
                        gt_boxes.extend(new_obj_box)
                        gt_classes.extend([gt_classes[obj] for i in range(k_num_box)])

                        for i in range(k_num_box):
                            relation.append(np.asarray([map[sub][i], map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([sub, map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([map[sub][i], obj, relation[rel][2]]))


                    elif (not map_keys.__contains__(sub) and map_keys.__contains__(obj)):
                        map[sub] = []
                        for i in range(k_num_box):
                            map[sub].append(len(gt_boxes) + i)
                        gt_boxes.extend(new_sub_box)
                        gt_classes.extend([gt_classes[sub] for i in range(k_num_box)])

                        for i in range(k_num_box):
                            relation.append(np.asarray([map[sub][i], map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([map[sub][i], obj, relation[rel][2]]))
                            relation.append(np.asarray([sub, map[obj][i], relation[rel][2]]))


                    else:
                        map[sub] = []
                        map[obj] = []
                        for i in range(k_num_box):
                            map[sub].append(len(gt_boxes) + i)
                            map[obj].append(len(gt_boxes) + k_num_box + i)
                        gt_boxes.extend(new_sub_box)
                        gt_boxes.extend(new_obj_box)
                        gt_classes.extend([gt_classes[sub] for i in range(k_num_box)])
                        gt_classes.extend([gt_classes[obj] for i in range(k_num_box)])

                        for i in range(k_num_box):
                            relation.append(np.asarray([map[sub][i], map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([sub, map[obj][i], relation[rel][2]]))
                            relation.append(np.asarray([map[sub][i], obj, relation[rel][2]]))



        dataset_vg.gt_boxes[id] = np.asarray(gt_boxes)
        dataset_vg.gt_classes[id] = np.asarray(gt_classes)
        dataset_vg.relationships[id] = np.asarray(relation)




    relationships = dataset_vg.relationships
    for relationship in tqdm(relationships):
        for re in range(len(relationship)):
            key=predicate_class_list[int(relationship[re][2])]
            dict_pred_num[key] = dict_pred_num.get(key,0)+1




    export_excel_direct(dict_pred_num,"/home/share/zhanghao/data/image/datasets/MBC-ATA/v1.xlsx")
    torch.save(dataset_vg.relationships,"/home/share/zhanghao/data/image/datasets/MBC-ATA/v1_relationships.pth")
    torch.save(dataset_vg.gt_classes, "/home/share/zhanghao/data/image/datasets/MBC-ATA/v1_gt_classes.pth")
    torch.save(dataset_vg.gt_boxes, "/home/share/zhanghao/data/image/datasets/MBC-ATA/v1_gt_boxes.pth")

    print("*************************Done*****************************")









if __name__ == "__main__":
     main()
