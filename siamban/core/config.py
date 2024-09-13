# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamban_r50_l234"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64


__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255
8 
__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

# __C.TRAIN.LOG_DIR = './logs'
# __C.TRAIN.LOG_DIR = '/home/master_110/n26092289/dataset/temp/logs'
__C.TRAIN.LOG_DIR = '/home/phd/n28111063/mml/HDD/n28111063/temp/logs'

# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/base_multi_task/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/base_no_salient_loss_multi_task/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/base_four_dataset/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/base_multi_task_4&5losses/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_base_multitask/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_base_multitask_salient_loss_3_4/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_base_multitask_salient_loss_3_4_epoch40/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/try/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/olny_multitask/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_onlymask/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_check_salient_loss/logs'

# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/real_mask_gt/logs'
# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/real_mask_gt_only_multitask/logs'

# __C.TRAIN.LOG_DIR = '/home/n26092289/storage_d/final_try/logs'





# __C.TRAIN.SNAPSHOT_DIR = './snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/master_110/n26092289/dataset/temp/snapshot'
__C.TRAIN.SNAPSHOT_DIR = '/home/phd/n28111063/mml/HDD/n28111063/temp/snapshot'

# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/base_multi_task/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/base_no_salient_loss_multi_task/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/base_four_dataset/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/base_multi_task_4&5losses/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_base_multitask/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_base_multitask_salient_loss_3_4/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_base_multitask_salient_loss_3_4_epoch40/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/try/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/olny_multitask/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_onlymask/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_check_salient_loss/snapshot'

# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/real_mask_gt/snapshot'
# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/real_mask_gt_only_multitask/snapshot'

# __C.TRAIN.SNAPSHOT_DIR = '/home/n26092289/storage_d/final_try/snapshot'



__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', 'GOT10K', 'LASOT')

__C.DATASET.VID = CN()
# __C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
# __C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.ROOT = 'dataset/share/VID/crop511'
__C.DATASET.VID.ANNO = 'dataset/share/VID/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
# __C.DATASET.VID.NUM_USE = 100000
########## 我加的程式碼 ########## 
__C.DATASET.VID.NUM_USE = 200000
########## 我加的程式碼 ########## 

__C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
# __C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.ROOT = 'dataset/share/ytb-crop511/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'dataset/share/ytb-crop511/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000

__C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
# __C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.ROOT = 'dataset/share/coco/crop511'
__C.DATASET.COCO.ANNO = 'dataset/share/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
# __C.DATASET.COCO.NUM_USE = 100000
########## 我加的程式碼 ########## 
__C.DATASET.COCO.NUM_USE = 117266
########## 我加的程式碼 ########## 

__C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
# __C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.ROOT = 'dataset/share/det/crop511'
__C.DATASET.DET.ANNO = 'dataset/share/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = 200000
########## 我加的程式碼 ########## 
__C.DATASET.DET.NUM_USE = 100000
########## 我加的程式碼 ########## 

__C.DATASET.GOT10K = CN()
# __C.DATASET.GOT10K.ROOT = 'training_dataset/got_10k/crop511'
# __C.DATASET.GOT10K.ANNO = 'training_dataset/got_10k/train.json'
__C.DATASET.GOT10K.ROOT = 'dataset/training_dataset/got_10k/crop511'
__C.DATASET.GOT10K.ANNO = 'dataset/training_dataset/got_10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 100
__C.DATASET.GOT10K.NUM_USE = 200000

__C.DATASET.LASOT = CN()
# __C.DATASET.LASOT.ROOT = 'training_dataset/lasot/crop511'
# __C.DATASET.LASOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LASOT.ROOT = 'dataset/lasot/crop511'
__C.DATASET.LASOT.ANNO = 'dataset/lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = 200000

########## 我加的程式碼 ########## 
__C.DATASET.YTB_VOS = CN()
__C.DATASET.YTB_VOS.ROOT = 'dataset/share/ytb-crop511/crop511'
__C.DATASET.YTB_VOS.ANNO = 'dataset/share/ytb-crop511/train.json'
__C.DATASET.YTB_VOS.FRAME_RANGE = 20
__C.DATASET.YTB_VOS.NUM_USE = 200000
########## 我加的程式碼 ########## 

# __C.DATASET.VIDEOS_PER_EPOCH = 1000000
########## 我加的程式碼 ########## 
__C.DATASET.VIDEOS_PER_EPOCH = 600000
########## 我加的程式碼 ########## 
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN'

__C.BAN.KWARGS = CN(new_allowed=True)

########## 我加的程式碼 ########## 
__C.MASK = CN()

__C.MASK.MASK = False

__C.MASK.TYPE = 'MaskCorr'

__C.MASK.KWARGS = CN(new_allowed=True)
########## 我加的程式碼 ########## 

# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamBANTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.14

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.45

# Interpolation learning rate
__C.TRACK.LR = 0.30

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5
