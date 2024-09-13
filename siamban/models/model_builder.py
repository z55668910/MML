# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck

########## 我加的程式碼 ########## 
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from PIL import Image


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


# def select_mask_logistic_loss(mask, weight, salient_0, salient_1, pred_mask, i ,o_sz=63, g_sz=127):
def select_mask_logistic_loss(mask, weight, salient_0, salient_1, pred_mask ,o_sz=63, g_sz=127):
    # print("weight0 : ", weight.shape) # (56, 1, 25, 25)
    weight = weight.view(-1)
    # print("weight1 : ", weight.shape) # (35000)
    pos = weight.data.eq(1).nonzero().squeeze() # calculate only where the element is positive
    if pos.nelement() == 0: return pred_mask.sum() * 0, pred_mask.sum() * 0, pred_mask.sum() * 0

    # print('salient_0.shape : ',salient_0.shape)
    if len(salient_0.shape) == 4:
        # print("0 : ", salient_0.shape) # ([56, 1, 625, 625])
        salient_0 = salient_0.permute(0, 2, 3, 1).contiguous().view(-1, 1, 25, 25)
        # print("1 : ", salient_0.shape) # ([35000, 1, 25, 25])
        # print("pos : ", pos)
        salient_0 = torch.index_select(salient_0, 0, pos)
        # print("2 : ", salient_0.shape)
        salient_0 = nn.UpsamplingBilinear2d(size=[127, 127])(salient_0)
        # print("3 : ", salient_0.shape)
        salient_0 = salient_0.view(-1, 127 * 127)
        # print("4 : ", salient_0.shape)
    else:
        salient_0 = torch.index_select(salient_0, 0, pos)
        # print('final_salient_0.shape : ',salient_0.shape)

    # print("salient_1.shape", salient_1.shape)
    if len(salient_1.shape) == 4:
        salient_1 = salient_1.permute(0, 2, 3, 1).contiguous().view(-1, 1, 25, 25)
        salient_1 = torch.index_select(salient_1, 0, pos)
        salient_1 = nn.UpsamplingBilinear2d(size=[127, 127])(salient_1)
        salient_1 = salient_1.view(-1, 127 * 127)
    else:
        salient_1 = torch.index_select(salient_1, 0, pos)

    # print("pred_mask.shape", pred_mask.shape)
    if len(pred_mask.shape) == 4:
        # print("1 : ", pred_mask.shape)
        pred_mask = pred_mask.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        # print("2 : ", pred_mask.shape)
        pred_mask = torch.index_select(pred_mask, 0, pos)
        # print("3 : ", pred_mask.shape)
        pred_mask = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(pred_mask)
        # print("4 : ", pred_mask.shape)
        pred_mask = pred_mask.view(-1, g_sz * g_sz)
        # print("5 : ", pred_mask.shape)
    else:
        pred_mask = torch.index_select(pred_mask, 0, pos)

 
    # print("mask",mask.shape)
    # print("mask",(mask[0, 0]) * 255)
    # temp = ((mask[0, 0]) * 255).cpu().detach().numpy()
    # cv2.imwrite('/home/n26092289/temp/mask'+str(i)+'.png', temp)  


    # print("mask:",mask.shape)
    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    # print("0_mask_uf:",mask_uf.shape)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
    # print("1_mask_uf:",mask_uf.shape)
    mask_uf = torch.index_select(mask_uf, 0, pos)
    # print("2_mask_uf:",mask_uf.shape)
    ########## 我加的程式碼 ########## 
    # mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
    # print("0_mask_uf:",mask_uf.shape)
    # mask_temp = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, 127 * 127)
    # print("1_mask_uf:",mask_temp.shape)
    # mask_temp = torch.index_select(mask_temp, 0, pos)
    # print("mask_temp:",mask_temp.shape)
    ########## 我加的程式碼 ########## 

    salient_0_loss = F.soft_margin_loss(salient_0, mask_uf)
    salient_1_loss = F.soft_margin_loss(salient_1, mask_uf)
    pred_mask_loss = F.soft_margin_loss(pred_mask, mask_uf)   

    return salient_0_loss, salient_1_loss, pred_mask_loss






# def select_mask_logistic_loss(mask, weight, pred_mask,o_sz=63, g_sz=127):
#     weight = weight.view(-1)
#     pos = weight.data.eq(1).nonzero().squeeze()
#     if pos.nelement() == 0: return pred_mask.sum() * 0, pred_mask.sum() * 0, pred_mask.sum() * 0

#     if len(pred_mask.shape) == 4:
#         # print("1 : ", pred_mask.shape) # ([28, 3969, 25, 25])
#         pred_mask = pred_mask.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
#         # print("2 : ", pred_mask.shape) # ([17500, 1, 63, 63])
#         pred_mask = torch.index_select(pred_mask, 0, pos)
#         # print("3 : ", pred_mask.shape) # ([175, 1, 63, 63])
#         pred_mask = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(pred_mask)
#         # print("4 : ", pred_mask.shape) # ([175, 1, 127, 127])
#         pred_mask = pred_mask.view(-1, g_sz * g_sz)
#         # print("5 : ", pred_mask.shape) # ([175, 16129])
#     else:
#         pred_mask = torch.index_select(pred_mask, 0, pos)

#     mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
#     mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
#     mask_uf = torch.index_select(mask_uf, 0, pos)

#     pred_mask_loss = F.soft_margin_loss(pred_mask, mask_uf)   

#     return pred_mask_loss




########## 我加的程式碼 ########## 

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        ########## 我加的程式碼 ########## 
        if cfg.MASK.MASK:
            self.mask_model = get_ban_head(cfg.MASK.TYPE,
                                        **cfg.MASK.KWARGS)


        # ResNet-50
        self.head_1 = conv3x3(256, 1, bias=True)
        self.head_2 = conv3x3(256, 1, bias=True)

        self.i = 0
        ########## 我加的程式碼 ########## 

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        # return {
        #         'cls': cls,
        #         'loc': loc
        #        }

        ########## 我加的程式碼 ########## 
        # cls, loc, mask = self.head(self.zf, xf)
        # return {
        #         'cls': cls,
        #         'loc': loc,
        #         'mask': mask
        #        }
    
        pred_mask = self.mask_model(self.zf[1], xf[1])

        return {
                'cls': cls,
                'loc': loc,
                'mask': pred_mask
               }

        ########## 我加的程式碼 ########## 




    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        ########## 我加的程式碼 ########## 
        mask_ground_truth = data['mask_ground_truth'].cuda()
        mask_weight = data['mask_weight'].cuda()
        temp_ori_search_img = data['temp_ori_search_img'].cuda()
        ########## 我加的程式碼 ########## 


        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)



        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.head(zf, xf)


        # temp = ((search[0, 0])).cpu().numpy()


        # temp = ((search[0])).cpu().numpy()
        # temp =  temp.transpose((1, 2, 0)).astype(np.float32)

        # cv2.imwrite('/home/n26092289/temp/search'+str(self.i)+'.png', temp)  

        # temp = ((template[0])).cpu().numpy()
        # temp =  temp.transpose((1, 2, 0)).astype(np.float32)

        # cv2.imwrite('/home/n26092289/temp/template'+str(self.i)+'.png', temp)  

        # temp = ((temp_ori_search_img[0])).cpu().numpy()
        # cv2.imwrite('/home/n26092289/temp/temp_ori_search_img'+str(self.i)+'.png', temp)  


        ########## 我加的程式碼 ########## 

        # pred_mask = self.mask_model(zf[2], xf[2])
        pred_mask = self.mask_model(zf[1], xf[1])

        search_shape = search.size()[2:]

        # print("xf[0] : ", xf[0].shape)
        # print("xf[1] : ", xf[1].shape)


        logits_2 = F.interpolate((self.head_1(xf[0])), size=[625, 625], mode='bilinear', align_corners=True)
        logits_3 = F.interpolate((self.head_2(xf[1])), size=[625, 625], mode='bilinear', align_corners=True)

        logits_2 = torch.sigmoid(logits_2)
        logits_3 = torch.sigmoid(logits_3)

        # temp = ((xf[0][0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/logits_2.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/xf[0]_'+str(self.i)+'.png', np.round(temp))  

        # temp = ((xf[1][0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/logits_3.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/xf[1]_'+str(self.i)+'.png', np.round(temp)) 

        # temp = ((logits_2[0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/logits_2.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/logits_2_'+str(self.i)+'.png', np.round(temp))  

        # temp = ((logits_3[0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/logits_3.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/logits_3_'+str(self.i)+'.png', np.round(temp))  


        # temp = ((pred_mask[0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/pred_mask.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/pred_mask_'+str(self.i)+'.png', np.round(temp))  


        # temp = ((mask_ground_truth[0, 0]) * 255).cpu().detach().numpy()
        # # cv2.imwrite('/home/n26092289/temp/mask_ground_truth.png', np.round(temp))  
        # cv2.imwrite('/home/n26092289/temp/mask_ground_truth.png'+str(self.i)+'.png', temp)  




        # print("logits_2 : ",logits_2.shape)

        # #這邊先註解
        # logits_2 = logits_2.view(-1, 51*51)
        # logits_3 = logits_3.view(-1, 51*51)
        # #這邊先註解

        salient_loss2, salient_loss3, mask_loss = select_mask_logistic_loss(mask_ground_truth, mask_weight, logits_2, logits_3, pred_mask)

        # salient_loss2= F.binary_cross_entropy_with_logits(logits_2, mask_ground_truth)
        # salient_loss3 = F.binary_cross_entropy_with_logits(logits_3, mask_ground_truth)

        # mask_loss = select_mask_logistic_loss(mask_ground_truth, mask_weight, pred_mask)


        # salient_loss2, salient_loss3, mask_loss = select_mask_logistic_loss(mask_ground_truth, mask_weight, logits_2, logits_3, pred_mask, self.i)
        # self.i= self.i+1




        ########## 我加的程式碼 ########## 


        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #     cfg.TRAIN.LOC_WEIGHT * loc_loss

        ########## 我加的程式碼 ##########

        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss  + 1/64 * salient_loss2 + 1/32 * salient_loss3 + 2 * mask_loss

        # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #     cfg.TRAIN.LOC_WEIGHT * loc_loss + 2 * mask_loss
        
        ########## 我加的程式碼 ########## 
        
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        
        ########### 我加的程式碼 ########## 
        outputs['salient_loss2'] = salient_loss2
        outputs['salient_loss3'] = salient_loss3
        outputs['mask_loss'] = mask_loss
        ########## 我加的程式碼 ########## 

        return outputs
