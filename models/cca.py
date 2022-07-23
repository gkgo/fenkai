""" code references: https://github.com/ignacio-rocco/ncnet and https://github.com/gengshan-y/VCN """

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=640, ratio=10):
        super(ChannelAttention, self).__init__()
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 80,1,5,5
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 80,1,5,5
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 80,1,5,5
        return self.sigmoid(x)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=10, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        self.pool_types = pool_types
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 50，640，1，1
                channel_att_raw = self.mlp(avg_pool)  # 50，640
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 50，640，1，1
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.relu(channel_att_sum).unsqueeze(2).unsqueeze(3)  # 50，640，1，1
        return scale

class match_block1(nn.Module):
    def __init__(self, inplanes):
        super(match_block1, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        self.ChannelGate = ChannelGate(self.in_channels)
        self.ChannelAttention = ChannelAttention(self.in_channels)
        self.SpatialAttention = SpatialAttention()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, spt, qry):
        # c_weight = self.ChannelGate(spt)
        # x1 = qry*c_weight + qry
        c_weight = self.ChannelAttention(spt)
        x = spt * c_weight
        x0 = self.SpatialAttention(x)
        x1 = x * x0 + qry
        return x1

class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        self.ChannelGate = ChannelGate(self.in_channels)
        self.ChannelAttention = ChannelAttention(self.in_channels)
        self.SpatialAttention = SpatialAttention()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, spt, qry):
        c_weight1 = self.ChannelAttention(spt)
        c_weight2 = self.ChannelAttention(qry)
        # c_weight = self.ChannelGate(spt)
        # x1 = qry*c_weight + qry
        xq = qry*c_weight1
        xs = spt*c_weight2
        xq0 = self.SpatialAttention(spt)
        xs0 = self.SpatialAttention(qry)
        x1 = xq * xq0 + qry
        x2 = xs * xs0 + spt
        return x2, x1

class CCA(torch.nn.Module):
    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()

        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv4d(in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        # because of the ReLU layers in between linear layers,
        # this operation is different than convolving a single time with the filters+filters^T
        # and therefore it makes sense to do this.
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        return x


class SepConv4d(nn.Module):
    """ approximates 3 x 3 x 3 x 3 kernels via two subsequent 3 x 3 x 1 x 1 and 1 x 1 x 3 x 3 """
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), ksize=3, do_padding=True, bias=False):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias, padding=0),
                nn.BatchNorm2d(out_planes))

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1, ksize, ksize),
                      stride=stride, bias=bias, padding=padding1),
            nn.BatchNorm3d(in_planes))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(ksize, ksize, 1),
                      stride=stride, bias=bias, padding=padding2),
            nn.BatchNorm3d(in_planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x
