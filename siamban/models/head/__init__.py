from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from siamban.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN

########## 我加的程式碼 ########## 
from siamban.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN, MaskCorr
########## 我加的程式碼 ########## 


BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN,
        'MaskCorr': MaskCorr
       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)

