from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor


@MODELS.register_module()
class FrustumRangePreprocessorPPT(BaseDataPreprocessor):
    """Frustum-Range Segmentor pre-processor for frustum region group.

    Args:
        H (int): Height of the 2D representation.
        W (int): Width of the 2D representation.
        fov_settings ({str: {"up": float, "down": float}}): Field-of-View of the sensor for each dataset.
        ignore_index (int): The label index to be ignored.
        num_contexts (int): The number of context being evaluated
        embedding_dim (int): The size of the context embeddings
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_settings: Dict[str, Dict[str, float]],
                 ignore_index: int,
                 num_contexts: int = 3,
                 embedding_dim: int = 256,
                 in_channels: int = 4,
                 non_blocking: bool = False) -> None:
        super(FrustumRangePreprocessorPPT,
              self).__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.in_channels = in_channels
        self.dataset_keys = list(fov_settings.keys())
        fov_down = torch.tensor([fov_settings[key]["down"] for key in self.dataset_keys], dtype=torch.float32)
        self.register_buffer('fov_down', fov_down / 180 * np.pi)
        fov_range = torch.tensor([abs(fov_settings[key]["down"]) + abs(fov_settings[key]["up"]) for key in self.dataset_keys], dtype=torch.float32)
        self.register_buffer('fov_range', fov_range / 180 * np.pi)
        self.ignore_index = ignore_index
        self.embedding = nn.Embedding(num_embeddings=num_contexts, embedding_dim=embedding_dim)
        self.embedding.weight.data.fill_(0)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform frustum region group based on ``BaseDataPreprocessor``.

        Args:
            data (dict): Data from dataloader. The dict contains the whole
                batch data.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        assert 'points' in inputs
        input_points = []
        input_ambients = []
        for points in inputs['points']:
            if points.shape[1] == self.in_channels:
                input_points.append(points)
                input_ambients.append(torch.zeros_like(points[:, -1]))
            elif points.shape[1] == self.in_channels + 1:
                input_points.append(points[:, :-1])
                input_ambients.append(points[:, -1])
            else:
                raise RuntimeError("Input cloud has more channels than in_channels!")

        batch_inputs['points'] = input_points
        batch_inputs['ambient'] = input_ambients

        assert 'context' in data
        batch_inputs['context'] = data['context']

        assert 'file_path' in data
        batch_inputs['file_path'] = data['file_path']

        voxel_dict = self.frustum_region_group(input_points, data['context'], data_samples)

        # add context embeddings to voxel_dict
        context_indices = [self.dataset_keys.index(context) for context in data['context']]
        indices = torch.tensor(context_indices, dtype=torch.int32, device=input_points[0].device, requires_grad=False)
        voxel_dict['context_vectors'] = self.embedding(indices)
        
        with torch.no_grad():
            voxel_dict['ambient'] = torch.cat(batch_inputs['ambient'], dim=0)
        
        batch_inputs['voxels'] = voxel_dict

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor], context: List[str],
                             data_samples: SampleList) -> dict:
        """Calculate frustum region of each point.

        Args:
            points (List[Tensor]): Point cloud in one data batch.

        Returns:
            dict: Frustum region information.
        """
        voxel_dict = dict()

        coors = []
        voxels = []

        for i, res in enumerate(points):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            context_index = self.dataset_keys.index(context[i])
            coors_y = 1.0 - (pitch - self.fov_down[context_index]) / self.fov_range[context_index]

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(
                coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(
                coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            voxels.append(res)

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                gt_semantic_segs = [sample.gt_pts_seg.pts_semantic_mask for sample in data_samples]
                voxel_dict['pts_semantic_mask'] = torch.cat(gt_semantic_segs, dim=0)

                import torch_scatter
                pts_semantic_mask = data_samples[
                    i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                res_voxel_coors, inverse_map = torch.unique(
                    res_coors, return_inverse=True, dim=0)
                voxel_semantic_mask = torch_scatter.scatter_mean(
                    F.one_hot(pts_semantic_mask).float(), inverse_map, dim=0)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label[res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
