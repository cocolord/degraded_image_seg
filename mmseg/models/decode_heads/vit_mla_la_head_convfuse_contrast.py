import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_
from ..losses import accuracy
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.vit import Block
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16, force_fp32
from mmseg.ops import resize
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

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

@HEADS.register_module()
class VIT_MLALAConvFuseContrastHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLALAConvFuseContrastHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.mlahead = MLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.fuse1 = nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.fuse2 = nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.cls = nn.Conv2d(self.mlahead_channels, self.num_classes, 3, padding=1)
        self.la_conv = Layer_Att()
        self.proj = ProjectionHead(256)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        # print('----mlahead---x.size()',x.size())
        # print('---input[4] ---',inputs[4].size())
        b,c,h,w = x.size()
        # print('---x before la---',x.size())
        # print('---x before fuse1---',x.size())
        x = self.fuse1(x)
        # print('---x before cat---',x.size())
        x = torch.cat((x,inputs[4]), dim=1)
        # print('---x after cat---',x.size())
        x = self.la_conv(x.view(b,2,-1,h,w))
        x_proj = self.proj(x)
        # print('---x after la---',x.size())
        x = self.fuse2(x)
        # y = x.cpu().detach().view(-1,x.size(1)).numpy()
        # np.save("/home/juan/Donglusen/Workspace/mmsegmentation/tests/tsne_embedding.npy", y)
        # print('---x after fuse2---',x.size())
        x = self.cls(x)
        # y = x.cpu().detach().view(-1,x.size(1)).numpy()
        # np.save("/home/juan/Donglusen/Workspace/mmsegmentation/tests/test_label.npy", y)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)    
        x_proj = F.interpolate(x_proj, size=self.img_size, mode='bilinear', align_corners=self.align_corners)    
        return x, x_proj

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, x_embed = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, x_embed)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]
    
    def losses(self, seg_logit, seg_label,embed=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        # print("seg_logit.shape", seg_logit.shape)
        # print("seg_label.shape", seg_label.shape)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            embed )
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss


@HEADS.register_module()
class VIT_MLALAConvFuseContrastMemHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128, 
                norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLALAConvFuseContrastMemHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.mlahead = MLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.fuse1 = nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.fuse2 = nn.Conv2d(2 * self.mlahead_channels, self.mlahead_channels, 3, 1, 1)
        self.cls = nn.Conv2d(self.mlahead_channels, self.num_classes, 3, padding=1)
        self.la_conv = Layer_Att()
        self.proj = ProjectionHead(256)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2], inputs[3])
        # print('----mlahead---x.size()',x.size())
        # print('---input[4] ---',inputs[4].size())
        b,c,h,w = x.size()
        # print('---x before la---',x.size())
        # print('---x before fuse1---',x.size())
        x = self.fuse1(x)
        # print('---x before cat---',x.size())
        x = torch.cat((x,inputs[4]), dim=1)
        # print('---x after cat---',x.size())
        x = self.la_conv(x.view(b,2,-1,h,w))
        x_proj = self.proj(x)
        # print('---x after la---',x.size())
        x = self.fuse2(x)
        # y = x.cpu().detach().view(-1,x.size(1)).numpy()
        # np.save("/home/juan/Donglusen/Workspace/mmsegmentation/tests/tsne_embedding.npy", y)
        # print('---x after fuse2---',x.size())
        x = self.cls(x)
        # y = x.cpu().detach().view(-1,x.size(1)).numpy()
        # np.save("/home/juan/Donglusen/Workspace/mmsegmentation/tests/test_label.npy", y)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)    
        x_proj = F.interpolate(x_proj, size=self.img_size, mode='bilinear', align_corners=self.align_corners)    
        return x, x_proj, queue

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, x_embed = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, x_embed)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]
    
    def losses(self, seg_logit, seg_label,embed=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        # print("seg_logit.shape", seg_logit.shape)
        # print("seg_label.shape", seg_label.shape)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            embed )
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
