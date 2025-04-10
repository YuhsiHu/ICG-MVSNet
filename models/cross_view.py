import torch
import torch.nn as nn
import numpy as np
from .module import *


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class CrossViewNet(nn.Module):

    def __init__(self, coarse_group=[8,8,4], fine_group=[8,4,4], coarse_depth=[8,8,4], fine_depth=[8,4,4]):
        super(CrossViewNet, self).__init__()
        self.upsample1 = Deconv2d(coarse_group[0]*coarse_depth[0], coarse_group[0]*coarse_depth[0], kernel_size=4, stride=2, padding=1)
        self.upsample2 = Deconv2d(coarse_group[1]*coarse_depth[1], coarse_group[1]*coarse_depth[1], kernel_size=4, stride=2, padding=1)
        self.upsample3 = Deconv2d(coarse_group[2]*coarse_depth[2], coarse_group[2]*coarse_depth[2], kernel_size=4, stride=2, padding=1)

        self.conv0_0c = nn.Sequential(Conv2d(coarse_group[0]*coarse_depth[0], coarse_depth[0], kernel_size=3, stride=1, padding=1),
                                      Conv2d(coarse_depth[0], fine_depth[0], kernel_size=1, stride=1, padding=0))
        
        self.conv0_1c = nn.Sequential(Conv2d(coarse_group[1]*coarse_depth[1], coarse_depth[1], kernel_size=3, stride=1, padding=1),
                                      Conv2d(coarse_depth[1], fine_depth[1], kernel_size=1, stride=1, padding=0))
        
        self.conv0_1f = nn.Sequential(Conv2d(fine_group[1]*fine_depth[1], fine_depth[1], kernel_size=3, stride=1, padding=1),
                                      Conv2d(fine_depth[1], fine_depth[1], kernel_size=1, stride=1, padding=0))
        
        self.conv0_2c = nn.Sequential(Conv2d(coarse_group[2]*coarse_depth[2], coarse_depth[2], kernel_size=3, stride=1, padding=1),
                                      Conv2d(coarse_depth[2], fine_depth[2], kernel_size=1, stride=1, padding=0))
        
        self.conv0_2f = nn.Sequential(Conv2d(fine_group[2]*fine_depth[2], fine_depth[2], kernel_size=3, stride=1, padding=1),
                                      Conv2d(fine_depth[2], fine_depth[2], kernel_size=1, stride=1, padding=0))

        self.output1 = nn.Conv2d((fine_group[0]+2)*fine_depth[0], (fine_group[0]+2)*fine_depth[0], kernel_size=1, stride=1, padding=0)
        self.output2 = nn.Conv2d((fine_group[1]+2)*fine_depth[1], (fine_group[1]+2)*fine_depth[1], kernel_size=1, stride=1, padding=0)
        self.output3 = nn.Conv2d((fine_group[2]+2)*fine_depth[2], (fine_group[2]+2)*fine_depth[2], kernel_size=1, stride=1, padding=0)

        # # ablation study
        # self.upsample1 = Deconv2d(coarse_group[0]*coarse_depth[0], coarse_group[0]*coarse_depth[0], kernel_size=4, stride=2, padding=1)
        # self.upsample2 = Deconv2d(coarse_group[1]*coarse_depth[1], coarse_group[1]*coarse_depth[1], kernel_size=4, stride=2, padding=1)
        # self.upsample3 = Deconv2d(coarse_group[2]*coarse_depth[2], coarse_group[2]*coarse_depth[2], kernel_size=4, stride=2, padding=1)

        # self.conv0_0c = nn.Sequential(Conv2d(coarse_group[0]*coarse_depth[0], coarse_depth[0]*2, kernel_size=3, stride=1, padding=1),
        #                               Conv2d(coarse_depth[0]*2, fine_depth[0]*2, kernel_size=1, stride=1, padding=0))
        
        # self.conv0_1c = nn.Sequential(Conv2d(coarse_group[1]*coarse_depth[1], coarse_depth[1]*2, kernel_size=3, stride=1, padding=1),
        #                               Conv2d(coarse_depth[1]*2, fine_depth[1]*2, kernel_size=1, stride=1, padding=0))
        
        # self.conv0_1f = nn.Sequential(Conv2d(fine_group[1]*fine_depth[1], fine_depth[1]*2, kernel_size=3, stride=1, padding=1),
        #                               Conv2d(fine_depth[1]*2, fine_depth[1]*2, kernel_size=1, stride=1, padding=0))
        
        # self.conv0_2c = nn.Sequential(Conv2d(coarse_group[2]*coarse_depth[2], coarse_depth[2]*2, kernel_size=3, stride=1, padding=1),
        #                               Conv2d(coarse_depth[2]*2, fine_depth[2]*2, kernel_size=1, stride=1, padding=0))
        
        # self.conv0_2f = nn.Sequential(Conv2d(fine_group[2]*fine_depth[2], fine_depth[2]*2, kernel_size=3, stride=1, padding=1),
        #                               Conv2d(fine_depth[2]*2, fine_depth[2]*2, kernel_size=1, stride=1, padding=0))

        # self.output1 = nn.Conv2d((fine_group[0]+4)*fine_depth[0], (fine_group[0]+4)*fine_depth[0], kernel_size=1, stride=1, padding=0)
        # self.output2 = nn.Conv2d((fine_group[1]+4)*fine_depth[1], (fine_group[1]+4)*fine_depth[1], kernel_size=1, stride=1, padding=0)
        # self.output3 = nn.Conv2d((fine_group[2]+4)*fine_depth[2], (fine_group[2]+4)*fine_depth[2], kernel_size=1, stride=1, padding=0)

    def forward(self, c, f, stage_idx):
        """        
        :param c: [B, {8,8,4}, {8,8,4}, H, W], coarse cost volume B G D H/2 W/2
        :param f: [B, {8,4,4}, {8,4,4}, H, W], fine cost volume   B G D H W
        """
        ori = f # B G D H W
        c = torch.flatten(c, 1, 2) # B G D H/2 W/2 -> B G*D H/2 W/2
        f = torch.flatten(f, 1, 2) # B G D H W -> B G*D H W
        # print("------------------")
        # print("ori c f: ", ori.shape, c.shape, f.shape)
        if stage_idx == 1:
            c = self.upsample1(c) # B G*D H/2 W/2 -> B G*D H W
            c2f = self.conv0_0c(c) # B D H W
            f2f = self.conv0_0c(f) # B D H W
            c2f = c2f.unsqueeze(1)
            f2f = f2f.unsqueeze(1)
            cf_cost_1 = torch.concat((c2f, f2f, ori),1) # B 1+1+G D H W
            cf_cost = torch.flatten(cf_cost_1, 1, 2)
            final_cost = self.output1(cf_cost).unsqueeze(1) # B 1 (1+1+G)*D H W
            final_cost = final_cost.view(ori.shape[0], cf_cost_1.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])
        elif stage_idx == 2:
            c = self.upsample2(c) # B G*D H/2 W/2 -> B G*D H W
            c2f = self.conv0_1c(c) # B 4 H W
            f2f = self.conv0_1f(f) # B 4 H W
            c2f = c2f.unsqueeze(1) # B 1 4 H W
            f2f = f2f.unsqueeze(1) # B 1 4 H W
            cf_cost_2 = torch.concat((c2f, f2f, ori),1) # B 1+1+G 4 H W
            cf_cost = torch.flatten(cf_cost_2, 1, 2) # B (1+1+G)*4 H W
            final_cost = self.output2(cf_cost).unsqueeze(1)
            final_cost = final_cost.view(ori.shape[0], cf_cost_2.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])
        elif stage_idx == 3:
            c = self.upsample3(c) # B G*D H/2 W/2 -> B G*D H W
            c2f = self.conv0_2c(c) # B 4 H W
            f2f = self.conv0_2f(f) # B 4 H W
            c2f = c2f.unsqueeze(1) # B 1 4 H W
            f2f = f2f.unsqueeze(1) # B 1 4 H W
            cf_cost_3 = torch.concat((c2f, f2f, ori),1) # B 1+1+G 4 H W
            cf_cost = torch.flatten(cf_cost_3, 1, 2) # B (1+1+G)*4 H W
            final_cost = self.output3(cf_cost).unsqueeze(1)
            final_cost = final_cost.view(ori.shape[0], cf_cost_3.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])

        return final_cost

    # # ablation study
    # def forward(self, c, f, stage_idx):
    #     """        
    #     :param c: [B, {8,8,4}, {8,8,4}, H, W], coarse cost volume B G D H/2 W/2
    #     :param f: [B, {8,4,4}, {8,4,4}, H, W], fine cost volume   B G D H W
    #     """
    #     ori = f # B G D H W
    #     B, G, D, H, W = f.shape
    #     c = torch.flatten(c, 1, 2) # B G D H/2 W/2 -> B G*D H/2 W/2
    #     f = torch.flatten(f, 1, 2) # B G D H W -> B G*D H W
    #     # print("------------------")
    #     # print("ori c f: ", ori.shape, c.shape, f.shape)
    #     if stage_idx == 1:
    #         c = self.upsample1(c) # B G*D H/2 W/2 -> B G*D H W
    #         c2f = self.conv0_0c(c) # B D*2 H W
    #         f2f = self.conv0_0c(f) # B D*4 H W
    #         c2f = c2f.view(B, 2, D, H, W)
    #         f2f = f2f.view(B, 2, D, H, W)
    #         cf_cost_1 = torch.concat((c2f, f2f, ori),1) # B 2+2+G D H W
    #         cf_cost = torch.flatten(cf_cost_1, 1, 2)
    #         final_cost = self.output1(cf_cost).unsqueeze(1) # B 1 (2+2+G)*D H W
    #         final_cost = final_cost.view(ori.shape[0], cf_cost_1.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])
    #     elif stage_idx == 2:
    #         c = self.upsample2(c) # B G*D H/2 W/2 -> B G*D H W
    #         c2f = self.conv0_1c(c) # B 4 H W
    #         f2f = self.conv0_1f(f) # B 4 H W
    #         c2f = c2f.view(B, 2, D, H, W)
    #         f2f = f2f.view(B, 2, D, H, W)
    #         cf_cost_2 = torch.concat((c2f, f2f, ori),1) # B 2+2+G 4 H W
    #         cf_cost = torch.flatten(cf_cost_2, 1, 2) # B (2+2+G)*4 H W
    #         final_cost = self.output2(cf_cost).unsqueeze(1)
    #         final_cost = final_cost.view(ori.shape[0], cf_cost_2.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])
    #     elif stage_idx == 3:
    #         c = self.upsample3(c) # B G*D H/2 W/2 -> B G*D H W
    #         c2f = self.conv0_2c(c) # B 4 H W
    #         f2f = self.conv0_2f(f) # B 4 H W
    #         c2f = c2f.view(B, 2, D, H, W)
    #         f2f = f2f.view(B, 2, D, H, W)
    #         cf_cost_3 = torch.concat((c2f, f2f, ori),1) # B 2+2+G 4 H W
    #         cf_cost = torch.flatten(cf_cost_3, 1, 2) # B (2+2+G)*4 H W
    #         final_cost = self.output3(cf_cost).unsqueeze(1)
    #         final_cost = final_cost.view(ori.shape[0], cf_cost_3.shape[1], ori.shape[2], ori.shape[3], ori.shape[4])

    #     return final_cost