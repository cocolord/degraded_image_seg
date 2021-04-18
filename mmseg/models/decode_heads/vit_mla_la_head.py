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
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B, N, C, H, W)
            returns :
                out : attention map + input feature (B, NC, H, W)
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out


@HEADS.register_module()
class VIT_MLALAHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLALAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels, self.num_classes, 3, padding=1)
        self.la = Layer_Att()

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        b,c,h,w = x.size()
        x = self.cls(self.la(x.view(b,4,-1,h,w)))
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)        
        return x
