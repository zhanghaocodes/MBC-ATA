# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
sys.path.append(r"/home/share/zhanghao/codes/MBC-ATA")

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

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
from tqdm import tqdm
import numpy as np
import random
# from maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors import resampleFlag
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

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


def train(cfg, local_rank, distributed, logger):
    best_epoch = 0
    best_mR = 0.0
    logger.info("***********************Step 1: loading models***********************")
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
 
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    logger.info("***********************Step 2: setting optimizer and shcedule***********************")
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0
    logger.info("***********************Step 2: over***********************")
    print('\n')

    output_dir = cfg.OUTPUT_DIR

    logger.info("***********************Step 3: loading pre-trained model***********************")
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        pretrain_object_detector_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT_VG
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        pretrain_object_detector_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT_GQA
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(pretrain_object_detector_dir,
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(pretrain_object_detector_dir, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')
    logger.info("***********************Step 3: over***********************")
    print('\n')

    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))


    # logger.info("***********************freeze parameter!!!!!!!!!!!!!***********************")
    # for p in model.roi_heads.relation.predictor.context_layer.parameters():
    #     p.requires_grad = False


    logger.info("***********************Step 4: preparing data***********************")
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    logger.info("***********************Step 4: over***********************")
    print('\n')

    if False:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    logger.info("***********************Step training starts***********************")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True

    if(cfg.ATA):
        logger.info("***********************USING ATA***********************")
        dataset_vg = train_data_loader.dataset
        lenth = len(dataset_vg.gt_boxes)
        for id in tqdm(range(0, lenth)):
            gt_boxes = list(dataset_vg.gt_boxes[id])
            gt_classes = list(dataset_vg.gt_classes[id])
            img_info = dataset_vg.img_info[id]
            relation = list(dataset_vg.relationships[id])
            map = {}
            for rel in range(len(relation)):  # 遍历每一个关系三元组
                pred = relation[rel][2]
                if (pred < 25):
                    continue
                else:
                    sub = relation[rel][0]
                    obj = relation[rel][1]
                    if (sub > len(gt_boxes) or obj > len(gt_boxes)):
                        continue

                    sub_box = gt_boxes[sub]
                    obj_box = gt_boxes[obj]

                    iou = 0.8
                    new_sub_box_list = createTranslationList(sub_box, iou, img_info)
                    new_sub_box_list.extend(createScaleList(sub_box, iou, img_info))

                    k_num_box = 1
                    new_sub_box = random.sample(new_sub_box_list, k_num_box)
                    # new_sub_box = [sub_box]

                    new_obj_box_list = createTranslationList(obj_box, iou, img_info)
                    new_obj_box_list.extend(createScaleList(obj_box, iou, img_info))
                    new_obj_box = random.sample(new_obj_box_list, k_num_box)
                    # new_obj_box = [obj_box]

                    if (not map):
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

                    else:
                        map_keys = map.keys()
                        if (map_keys.__contains__(sub) and map_keys.__contains__(obj)):
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


        train_data_loader.dataset.relationships = dataset_vg.relationships
        train_data_loader.dataset.gt_classes = dataset_vg.gt_classes
        train_data_loader.dataset.gt_boxes = dataset_vg.gt_boxes

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            if val_result > best_mR:
                best_epoch = iteration
                best_mR = val_result
            logger.info("now best epoch in mR@k is : %d, num is %.4f" % (best_epoch, best_mR))
            logger.info("Validation Result: %.4f" % val_result)
 
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    logger.info("***********************Step training over***********************")

    name = "model_{:07d}".format(best_epoch)
    last_filename = os.path.join(cfg.OUTPUT_DIR, "{}.pth".format(name))
    output_folder = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    with open(output_folder, "w") as f:
        f.write(last_filename)
    print('\n\n')
    logger.info("Best Epoch is : %.4f" % best_epoch)

    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()
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
        dataset_names = cfg.DATASETS.VG_VAL
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        dataset_names = cfg.DATASETS.GQA_200_VAL
    else:
        dataset_names = None
        exit('wrong Dataset name!')

    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    #torch.cuda.empty_cache()
    return val_result

def run_test(cfg, model, distributed, logger, is_best=False):
    if is_best:
        logger.info("***********************Best testing starts***********************")
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
    else:
        logger.info("***********************Step testing starts***********************")
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()
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

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            if is_best:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_best", dataset_name)
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_final", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()
        logger.info("***********************Step testing over***********************")
        print('\n\n')


def main():
    # torch.manual_seed(3407)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        #default="/home/share/zhanghao/codes/Motif_Codebase/configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml",
        default="/home/share/zhanghao/codes/MBC-ATA/configs/MBC-ATA.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    print('\n\n\n\n')
    logger.info("---------------------------new training!---------------------------")
    logger.info("***********************Step 0: loading configs***********************")
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    logger.info("***********************Step 0: over***********************")
    print('\n')


    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    main()
