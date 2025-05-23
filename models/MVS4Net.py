import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, time
from models.module import *
from .cross_view import CrossViewNet


class MVS4net(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8, 
                reg_channel=8, stage_splits=[8,8,4,4], depth_interals_ratio=[0.5,0.5,0.5,0.5],
                group_cor=True, group_cor_dim=[8,8,4,4],
                inverse_depth=True,
                agg_type='ConvBnReLU3D',
                attn_temp=2,
                attn_fuse_d=True
                ):
        super(MVS4net, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False)
        self.stagenet = stagenet(inverse_depth, attn_fuse_d, attn_temp)
        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        if reg_net == 'reg3d':
            self.down_size = [3,3,2,2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d':
                # self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
                if idx == 0:
                    self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
                else:
                    self.reg.append(reg2d(input_channel=in_dim + 2, base_channel=reg_channel + 2, conv_name=agg_type))
                    # # ablation study
                    # self.reg.append(reg2d(input_channel=in_dim + 4, base_channel=reg_channel + 4, conv_name=agg_type))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=1, base_channels=reg_channel, down_size=self.down_size[idx]))
        
        self.cross_view_net = CrossViewNet(coarse_group=self.group_cor_dim[:3], fine_group=self.group_cor_dim[1:4], coarse_depth=self.stage_splits[:3], fine_depth=self.stage_splits[1:4])

        
    def forward(self, imgs, proj_matrices, depth_values):
        # start_time = time.time()
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()

        ref_outputs, src_outputs = self.feature(imgs)

        # step 2. iter (multi-scale)
        outputs = {}
        coarse_cost = None
        for stage_idx in range(self.num_stage):
            ref_features_stage = ref_outputs["stage{}".format(stage_idx + 1)]
            src_features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in src_outputs]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B, C, H, W = ref_features_stage.shape
            
            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device, imgs[0][0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device, imgs[0][0].dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_interval = (depth_max - depth_min) / 192 
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx], self.depth_interals_ratio[stage_idx] * depth_interval, H, W)
            
            outputs_stage = self.stagenet(ref_features_stage, src_features_stage, proj_matrices_stage, depth_hypo=depth_hypo, cross_view_net=self.cross_view_net, regnet=self.reg[stage_idx], coarse_cost=coarse_cost, stage_idx=stage_idx,
                                    group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                    split_itv=self.depth_interals_ratio[stage_idx])
            
            coarse_cost = outputs_stage['coarse_cost']
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        # end_time = time.time()
        # print("forward time: ", end_time - start_time)
        return outputs


def cross_entropy_loss(mask_true,hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape
    attn_weight = torch.where(attn_weight <= 0.0, torch.zeros_like(attn_weight), attn_weight)
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6
    gt_index_image = torch.argmin(torch.abs(hypo_depth-depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
    gt_index_volume = torch.zeros(B, D, H, W).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) 
    
    return masked_cross_entropy


def cross_entropy_loss_w(mask_true,hypo_depth, depth_gt, attn_weight):
    # 计算GT深度到相邻两个深度假设的距离
    _, indices = torch.topk(torch.abs(hypo_depth - depth_gt.unsqueeze(1)), k=2, dim=1, largest=False) # B 2 H W
    d1_index = indices[:, 0, :, :]  # B H W
    d2_index = indices[:, 1, :, :]  # B H W
    d1_depth = torch.gather(hypo_depth, 1, d1_index.unsqueeze(1)) # B 1 H W
    d2_depth = torch.gather(hypo_depth, 1, d2_index.unsqueeze(1)) # B 1 H W
    distance_d1 = torch.abs(d1_depth - depth_gt.unsqueeze(1))  # B 1 H W
    distance_d2 = torch.abs(d2_depth - depth_gt.unsqueeze(1))  # B 1 H W
    w1 = distance_d2 / (distance_d1 + distance_d2 + 1e-6) # B 1 H W
    w2 = distance_d1 / (distance_d1 + distance_d2 + 1e-6) # B 1 H W
    
    B, D, H, W = attn_weight.shape
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6
    gt_index_volume = torch.zeros_like(attn_weight)
    d1_index_image = torch.mul(mask_true, d1_index.type(torch.float))
    d1_index_image = torch.round(d1_index_image).type(torch.long).unsqueeze(1) # B 1 H W
    d2_index_image = torch.mul(mask_true, d2_index.type(torch.float))
    d2_index_image = torch.round(d2_index_image).type(torch.long).unsqueeze(1) # B 1 H W
    gt_index_volume.scatter_(1, d1_index_image, w1) # B D H W
    gt_index_volume.scatter_(1, d2_index_image, w2) # B D H W
    
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) 
    
    return masked_cross_entropy


def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        # depth_minmax = stage_inputs['depth_minmax']
        # depth_indices = stage_inputs['depth_indices']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key] # B H W

        # mask range
        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= 0.5 * depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())
        # cross-entropy 
        this_stage_ce_loss = cross_entropy_loss(mask,hypo_depth, depth_gt, attn_weight)
        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss * stage_lw[stage_idx]

    return total_loss, stage_ce_loss, range_err_ratio


def Blend_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy 
        this_stage_ce_loss = cross_entropy_loss(mask,hypo_depth, depth_gt, attn_weight)
      
        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss

    depth_interval = hypo_depth[:,0,:,:]-hypo_depth[:,1,:,:]
   
    abs_err = torch.abs(depth_gt[mask] - depth_pred[mask])
    abs_err_scaled = abs_err /(depth_interval[mask]*192./128.)
    epe = abs_err_scaled.mean()
    err3 = (abs_err_scaled<=3).float().mean()
    err1= (abs_err_scaled<=1).float().mean()
    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1