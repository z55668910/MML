from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.xcorr import xcorr_fast, xcorr_depthwise

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)
        ########## 我加的程式碼 ########## 
        # self.mask = DepthwiseXCorr(in_channels, out_channels, 51*51)
        ########## 我加的程式碼 ########## 
        
    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

        ########## 我加的程式碼 ########## 
        # mask = self.mask(z_f, x_f)
        # return cls, loc, mask
        ########## 我加的程式碼 ########## 


class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
            ########## 我加的程式碼 ##########  
            # self.mask_weight = nn.Parameter(torch.ones(len(in_channels)))
            ########## 我加的程式碼 ########## 
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        ########## 我加的程式碼 ##########  
        # mask = []
        ########## 我加的程式碼 ##########  

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box'+str(idx))
            c, l = box(z_f, x_f)
            ########## 我加的程式碼 ##########  
            # c, l, m = box(z_f, x_f)
            # mask.append(m)
            ########## 我加的程式碼 ##########  
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            ########## 我加的程式碼 ##########  
            # mask_weight = F.softmax(self.mask_weight, 0)
            ########## 我加的程式碼 ##########  

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)

        # ########## 我加的程式碼 ########## 
        # if self.weighted:
        #     return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight), weighted_avg(mask, mask_weight)
        # else:
        #     return avg(cls), avg(loc), avg(mask)
        # ########## 我加的程式碼 ########## 

########## 我加的程式碼 ##########  
class MaskCorr(BAN):
    def __init__(self, in_channels, cls_out_channels, oSz=63, weighted=False):
        super(MaskCorr, self).__init__()
        # self.weighted = weighted
        self.oSz = oSz
        self.mask = DepthwiseXCorr(256, 256, 63*63)

        # for i in range(len(in_channels)):
        #     self.add_module('mask_box'+str(i+2), DepthwiseXCorr(in_channels[i], in_channels[i], 63*63))
        # if self.weighted:
        #     self.mask_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z, x):
        return self.mask(z, x)

    # def forward(self, z_fs, x_fs):
    #     mask = []
    #     for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
    #         box = getattr(self, 'mask_box'+str(idx))
    #         m = box(z_f, x_f)
    #         mask.append(m)

    #     if self.weighted:
    #         mask_weight = F.softmax(self.mask_weight, 0)

    #     def avg(lst):
    #         return sum(lst) / len(lst)

    #     def weighted_avg(lst, weight):
    #         s = 0
    #         for i in range(len(weight)):
    #             s += lst[i] * weight[i]
    #         return s

    #     if self.weighted:
    #         return weighted_avg(mask, mask_weight)
    #     else:
    #         return avg(mask)


########## 我加的程式碼 ##########  
