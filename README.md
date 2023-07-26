# MBC-ATA: Maximum Binary Classification and Anchor-based Triplet Augmentation for Unbiased Scene Graph Generation

This repository contains the code for our paper MBC-ATA: Maximum Binary Classification and Anchor-based Triplet Augmentation for Unbiased Scene Graph Generation.

## Installation

Check [INSTALL.md](https://github.com/zhanghaocodes/MBC-ATA/blob/master/INSTALL.md) for installation instructions, the recommended configuration is cuda-10.1 & pytorch-1.6.

## Dataset

The following is adapted from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`.
2. Download the [scene graphs](https://1drv.ms/u/s!AjK8-t5JiDT1kxyaarJPzL7KByZs?e=bBffxj) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catalog.py`.

## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR).

## Perform training on Scene Graph Generation

### Set the dataset path

First, please refer to the `SHA_GCL_extra/dataset_path.py` and set the `datasets_path` to be your dataset path, and organize all the files like this:

```
datasets
  |-- vg
    |--detector_model
      |--pretrained_faster_rcnn
        |--model_final.pth   
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    

```



### Perform some training configurations in the   `configs/MBC-ATA.yaml` 

#### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes, and pairwise relationships.

For **Predicate Classification (PredCls)**, you need to set:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```

For **Scene Graph Classification (SGCls)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

For **Scene Graph Detection (SGDet)**:

```
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

#### Choose your model

We abstract various SGG models to be different `relation-head predictors` in the file `roi_heads/relation_head/roi_relation_predictors.py`, which are independent of the Faster R-CNN backbone and relation-head feature extractor. You can use `GLOBAL_SETTING.RELATION_PREDICTOR` to select one of them:

```
GLOBAL_SETTING.RELATION_PREDICTOR 'MotifsLike_MBC'
```

#### Choose ATA strategy 

```
ATA: True
```



#### Choose your Encoder (For "MotifsLike" and "TransLike")

You need to further choose an object/relation encoder for "MotifsLike" or "TransLike" predictor, by setting the following parameter:

```
GLOBAL_SETTING.BASIC_ENCODER 'Self-Attention'
GLOBAL_SETTING.BASIC_ENCODER 'Motifs'
```

## Example of the Training Command

```
python tools/relation_train_net.py
```



## Acknowledgment

Our code is on top of [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG), we sincerely thank them for their well-designed codebase.