import torch
import torch.nn as nn
import numpy as np

from ..builder import HEADS
from .decode_head import BaseDecodeHead

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

if __name__ == '__main__':
    x = torch.rand(16, 2, 64, 32, 32)
    lam = Layer_Att()
    print(lam(x).shape)
