import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.vit import Block
from mmcv.cnn import build_norm_layer
from .vit_mla_head import MLAHead

class Layer_Att(nn.Module):
    def __init__(self):
        super(Layer_Att, self).__init__()
        self.gamma = nn.Parameter(torch.randn(1,requires_grad=True))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B, N, C, H, W)
            returns :
                out : attention map + input feature (B, NC, H, W)
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, N, -1)
        proj_key = x.contiguous().view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)#b x n x n
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.contiguous().view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.contiguous().view(m_batchsize, -1, height, width)
        return out


@HEADS.register_module()
class VIT_MLAConvFuseHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAConvFuseHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.mlahead = MLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.fuse1 = nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.fuse2 = nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.cls = nn.Conv2d(self.mlahead_channels, self.num_classes, 3, padding=1)
        # self.la = Layer_Att()
        # self.la_conv = Layer_Att()
        self.test_low_level_logit = True
    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        # print('----mlahead---x.size()',x.size())
        # print('---input[4] ---',inputs[4].size())
        b,c,h,w = x.size()
        # print('---x before la---',x.size())
        # x = self.la(x.view(b,4,-1,h,w))
        # print('---x before fuse1---',x.size())
        x = self.fuse1(x)
        # print('---x before cat---',x.size())
        x = torch.cat((x,inputs[4]), dim=1)
        # print('---x after cat---',x.size())
        # x = self.la_conv(x.view(b,2,-1,h,w))
        # print('---x after la---',x.size())
        x = self.fuse2(x)
        # print('---x after fuse2---',x.size())
        if self.test_low_level_logit:
            x = self.cls(inputs[4])
            return F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        x = self.cls(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)        
        return x
