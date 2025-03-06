from typing import List, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from ..normalization.normalization import PPTPointNorm, SequentialPassthrough, build_norm_layer_ppt
from mmdet3d.utils import ConfigType
from torch import Tensor
import numpy as np


class FinalClassifer(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_classes: int,
                 final_channels: Sequence[int],
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 alpha: float = 1.0,
                 use_ambient: bool = True,
                 ambient_out_channels: int = 8):
        super(FinalClassifer, self).__init__()
        
        self.alpha = alpha
        self.mlps = nn.ModuleList()
        self.use_ambient = use_ambient
        self.ambient_lin = nn.Linear(1, ambient_out_channels)
        in_channels += ambient_out_channels # Add additional channel for ambient

        for i in range(len(final_channels)):
            out_channels = final_channels[i]
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    build_norm_layer(norm_cfg["base_norm"], out_channels)[1],
                    nn.ReLU(inplace=True)))
            in_channels = out_channels
        self.lin = nn.Linear(final_channels[-1], out_classes, bias=True)
        self.num_classes = out_classes

    def manifold_mixup(self, features, voxel_dict: dict):
        # skip if alpha is <= 0
        if (self.alpha > 0):
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            return
        
        num_feat = features.shape[0]
        labels = voxel_dict["pts_semantic_mask"]
        assert(labels.shape[0] == num_feat), "pts_semantic_mask must match features length"
        index = torch.randperm(num_feat).cuda()
        mixed_features = lam * features + (1 - lam) * features[index,:]
        mixed_labels = lam * labels + (1 - lam) * labels[index,:]
        voxel_dict["pts_semantic_mask"] = mixed_labels
        return mixed_features, voxel_dict

    def forward(self, features, voxel_dict: dict, testing: bool = False):
        # determine mixup layer (set to negative to skip mixup)
        if not testing and self.alpha > 0.0 and "pts_semantic_mask" in voxel_dict:
            voxel_dict["pts_semantic_mask"] = nn.functional.one_hot(voxel_dict["pts_semantic_mask"], num_classes=self.num_classes).float()
            mixup_layer = torch.randint(0, len(self.mlps) + 1, (1,)).item()
        else:
            mixup_layer = -1
        
        if self.use_ambient:
            x = torch.cat((features,  self.ambient_lin(voxel_dict['ambient'].view(-1, 1))), dim=1) 
        else:
            x = torch.cat((features,  self.ambient_lin(torch.zeros_like(voxel_dict['ambient'].view(-1, 1)))), dim=1) 
        # apply each mlp in sequence
        for i, mlp in enumerate(self.mlps):
            # if current layer is chosen for mixup, apply mixup
            if (mixup_layer == i):
                x, voxel_dict = self.manifold_mixup(x, voxel_dict)
            # apply mixup
            x = mlp(x)
        # apply mixup if needed prior to final layer
        if (mixup_layer == len(self.mlps)):
            x, voxel_dict = self.manifold_mixup(x, voxel_dict)
        logits = self.lin(x)
        return logits, voxel_dict


@MODELS.register_module()
class FRHeadPPT(Base3DDecodeHead):

    def __init__(self,
                 in_channels: int,
                 middle_channels: Sequence[int],
                 final_channels: Sequence[int],
                 mixup_alpha: float = 0.0,
                 use_ambient: bool = True,
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 **kwargs) -> None:
        super(FRHeadPPT, self).__init__(**kwargs)

        self.loss_ce = MODELS.build(loss_ce)
        self.mixup_alpha = mixup_alpha

        self.mlps = nn.ModuleList()
        for i in range(len(middle_channels)):
            out_channels = middle_channels[i]
            self.mlps.append(
                SequentialPassthrough([
                    (1, nn.Linear(in_channels, out_channels, bias=False)),
                    (3, build_norm_layer_ppt(norm_cfg.copy(), out_channels)[1]),
                    (1, nn.ReLU(inplace=True))]))
            in_channels = out_channels
        
        self.final_classifier = FinalClassifer(middle_channels[-1], self.num_classes, final_channels, norm_cfg.copy(), mixup_alpha, use_ambient)

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return None

    def forward(self, voxel_dict: dict, testing: bool = False) -> dict:
        context_vectors = voxel_dict['context_vectors']
        # point_feats_backbone are the final point-wise multi-resolution outputs of the backbone
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]
        # point_feats are the initial input points features
        point_feats = voxel_dict['point_feats'][:-1]
        # voxel_feats are the voxel features fused at the final layer
        voxel_feats = voxel_dict['voxel_feats'][0]
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        pts_coors = voxel_dict['coors']
        # map_point_feats are the voxel features mapped to each point
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1],
                                      pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats, context_vectors, pts_coors)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]
        
        if self.dropout is not None:
            map_point_feats = self.dropout(map_point_feats)
        
        seg_logit, voxel_dict = self.final_classifier(map_point_feats, voxel_dict, testing)
        # seg_logit = self.cls_seg(map_point_feats)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs, dim=0)
    
    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        seg_logit = voxel_dict['seg_logit']
        if (self.mixup_alpha > 0.0) and ("pts_semantic_mask" in voxel_dict):
            seg_label = voxel_dict["pts_semantic_mask"]
        else:
            seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict, testing=True)

        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)

        final_seg_pred_list = []
        for seg_pred, input_metas in zip(seg_pred_list, batch_input_metas):
            if 'num_points' in input_metas:
                num_points = input_metas['num_points']
                seg_pred = seg_pred[:num_points]
            final_seg_pred_list.append(seg_pred)
        return final_seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        seg_logits = voxel_dict['seg_logit']

        coors = voxel_dict['coors']
        seg_pred_list = []
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            seg_pred_list.append(seg_logits[batch_mask])
        return seg_pred_list
