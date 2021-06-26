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
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)#b x n x n
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
class VisionTransformerUpHeadConvFuseLA(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, embed_dim=1024, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None, 
                num_conv=1, upsampling_method='bilinear', num_upsampe_layer=1, **kwargs):
        super(VisionTransformerUpHeadConvFuseLA, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.num_conv = num_conv
        self.norm = norm_layer(embed_dim)
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer
        self.conv_fuse = nn.Conv2d(384,256,3,1,1)
        out_channel=self.num_classes

        if self.num_conv==2:
            self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(256, out_channel, 1, 1)
            _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)

        elif self.num_conv==4:
            self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_4 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1)

            _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_1 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_2 = build_norm_layer(self.norm_cfg, 256)
            _, self.syncbn_fc_3 = build_norm_layer(self.norm_cfg, 256)
        
        self.la = Layer_Att()
        # Segmentation head

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        y = x[1]
        x = self._transform_inputs(x[0])
        if x.dim()==3:
            if x.shape[1] % 48 !=0:
                x = x[:,1:]
            x = self.norm(x)

        if self.upsampling_method=='bilinear':
            if x.dim()==3:
                n, hw, c = x.shape
                h=w = int(math.sqrt(hw))
                x = x.transpose(1,2).reshape(n, c, h, w)

            if self.num_conv==2:
                if self.num_upsampe_layer==2:
                    x = self.conv_0(x)
                    x = torch.cat([x,y],dim=1)
                    n,c,h,w = x.shape
                    x = self.la(x.view(n,2,-1,h,w))
                    x = x.view(n,-1,h,w)
                    x = self.conv_fuse(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
                elif self.num_upsampe_layer==1:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = self.conv_1(x)
                    x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_conv==4:
                if self.num_upsampe_layer==4:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = self.syncbn_fc_1(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_2(x)
                    x = torch.cat([x,y],dim=1)
                    n,c,h,w = x.shape
                    x = self.la(x.view(n,2,-1,h,w))
                    x = x.view(n,-1,h,w)
                    x = self.conv_fuse(x)
                    x = self.syncbn_fc_2(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_3(x)
                    x = self.syncbn_fc_3(x)
                    x = F.relu(x,inplace=True)
                    x = self.conv_4(x)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        return x

