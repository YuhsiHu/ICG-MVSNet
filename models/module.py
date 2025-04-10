import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np
from .intra_view import IntraViewNet
import time

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs, Ws = src_fea.shape[-2:]
    B, num_depth, Hr, Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:, :, d], grid.reshape(B, num_depth, Hr, Wr, 2)[:, d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                requires_grad=False).reshape(1, -1) * new_interval.squeeze(1)) #(B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) #(B, D, H, W)
    return depth_range_samples

def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv

    return 1./inverse_depth_hypo

def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H//2, W//2)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:, None, None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:, None, None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D'):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)

class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x.squeeze(1)  # B D H W


class FPN4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, gn=False):
        super(FPN4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels * 1, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

        # intra-view
        # self.intraview1 = nn.Sequential(
        #                     IntraViewNet5(32, 32),
        #                     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #                 )
        # self.intraview2 = nn.Sequential(
        #                     IntraViewNet5(16, 16),
        #                     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #                 )
        # self.intraview3 = nn.Sequential(
        #                     IntraViewNet5(8, 8),
        #                     # nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #                     # nn.BatchNorm2d(64),
        #                 )

        self.intraview1 = IntraViewNet(64, 64)
        self.intraview2 = IntraViewNet(64, 64)
        self.intraview3 = IntraViewNet(64, 64)


    def forward(self, imgs):
        ref_img, src_imgs = imgs[0], imgs[1:]
        device = ref_img.device

        ref_outputs, src_outputs = [],[]

        ref_conv0 = self.conv0(ref_img)
        ref_conv1 = self.conv1(ref_conv0)
        ref_conv2 = self.conv2(ref_conv1)
        ref_conv3 = self.conv3(ref_conv2)

        # intra = ref_conv3
        # ref_out1 = self.out1(intra)
        # intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(ref_conv2)
        # ref_out2 = self.out2(intra)
        # intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(ref_conv1)
        # ref_out3 = self.out3(intra)
        # intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(ref_conv0)
        # ref_out4 = self.out4(intra)

        ############## intra-view
        intra = ref_conv3
        ref_out1 = self.out1(intra) # B 64 H/8 W/8
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview1(self.inner1(ref_conv2))
        ref_out2 = self.out2(intra) # B 32 H/4 W/4
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview2(self.inner2(ref_conv1))
        ref_out3 = self.out3(intra) # B 16 H/2 W/2
        intra = F.interpolate(intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview3(self.inner3(ref_conv0))
        ref_out4 = self.out4(intra) # B 8 H W
        ##############

        ref_outputs = {}
        ref_outputs["stage1"] = ref_out1
        ref_outputs["stage2"] = ref_out2
        ref_outputs["stage3"] = ref_out3
        ref_outputs["stage4"] = ref_out4

        for src_idx, src_img in enumerate(src_imgs):
            src_out ={}
            src_conv0 = self.conv0(src_img)
            src_conv1 = self.conv1(src_conv0)
            src_conv2 = self.conv2(src_conv1)
            src_conv3 = self.conv3(src_conv2)
            src_intra = src_conv3.clone()

            # src_out1 = self.out1(src_intra)
            # src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(src_conv2)
            # src_out2 = self.out2(src_intra)
            # src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(src_conv1)
            # src_out3 = self.out3(src_intra)
            # src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(src_conv0)
            # src_out4 = self.out4(src_intra)

            ############## intra-view
            src_out1 = self.out1(src_intra) # B 64 H/8 W/8
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview1(self.inner1(src_conv2))
            src_out2 = self.out2(src_intra) # B 32 H/4 W/4
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview2(self.inner2(src_conv1))
            src_out3 = self.out3(src_intra) # B 16 H/2 W/2
            src_intra = F.interpolate(src_intra, scale_factor=2, mode="bilinear", align_corners=True) + self.intraview3(self.inner3(src_conv0))
            src_out4 = self.out4(src_intra) # B 8 H W
            ##############
            
            src_out["stage1"] = src_out1
            src_out["stage2"] = src_out2
            src_out["stage3"] = src_out3
            src_out["stage4"] = src_out4
            src_outputs.append(src_out)

        return ref_outputs, src_outputs


class stagenet(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2):
        super(stagenet, self).__init__()
        self.inverse_depth = inverse_depth
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp

    def forward(self, ref_feature, src_features, proj_matrices, depth_hypo, cross_view_net, regnet, coarse_cost, stage_idx, group_cor=False, group_cor_dim=8, split_itv=1):
        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        ref_volume =  ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W
            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src)**2 # B C D H W 
            del warped_src, src_proj, src_fea

            if not self.attn_fuse_d:
                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
            else:
                cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                cor_weight_sum += cor_weight  # B D H W
                cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat
        if not self.attn_fuse_d:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        del cor_weight_sum, src_features
        
        # step 3. regularization and cross-view
        if stage_idx == 0:
            coarse_cost = cor_feats # B G D H W
            attn_weight = regnet(cor_feats)  # B D H W
        elif stage_idx == 1:
            cost_after = cross_view_net(coarse_cost, cor_feats, stage_idx)
            coarse_cost = cor_feats
            attn_weight = regnet(cost_after)  # B D H W
        elif stage_idx == 2:
            cost_after = cross_view_net(coarse_cost, cor_feats, stage_idx)
            coarse_cost = cor_feats
            attn_weight = regnet(cost_after)  # B D H W
        else:
            cost_after = cross_view_net(coarse_cost, cor_feats, stage_idx)
            attn_weight = regnet(cost_after) # B D H W
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W
        ####################################################

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W

        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=2**(3-stage_idx), mode='bilinear', align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_feature.device, requires_grad=False)
        
        ret_dict = {"depth": depth,  "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo, "attn_weight": attn_weight, "coarse_cost": coarse_cost.detach()}
        
        if self.inverse_depth:
            last_depth_itv = 1./depth_hypo[:,2,:,:] - 1./depth_hypo[:,1,:,:] # B H W
            inverse_min_depth = 1/depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1/depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth
            
        return ret_dict