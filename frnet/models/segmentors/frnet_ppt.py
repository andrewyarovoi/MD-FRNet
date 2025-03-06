from typing import Dict

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from ...metrics.get_closest_dir import find_closest_directory

import os
import numpy as np
import re
from pypcd4 import PointCloud

@MODELS.register_module()
class FRNetPPT(EncoderDecoder3D):
    """Frustum-Range Segmentor.

    Args:
        voxel_encoder (dict or :obj:`ConfigDict`): The config for the voxel
            encoder of segmentor.
        backbone (dict or :obj:`ConfigDict`): The config for the backbone of
            segmentor.
        decode_head (dict or :obj:`ConfigDict`): The config for the decode head
            of segmentor.
        neck (dict or :obj:`ConfigDict`, optional): The config for the neck of
            segmentor. Defaults to None.
        auxiliary_head (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (dict or :obj:`ConfigDict`, optional): The config for
            training. Defaults to None.
        test_cfg (dict or :obj:`ConfigDict`, optional): The config for testing.
            Defaults to None.
        data_preprocessor (dict or :obj:`ConfigDict`, optional): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`],
            optional): The weight initialized config for :class:`BaseModule`.
            Defaults to None.
    """

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 log_path: str = None,
                 save_output: bool = False,
                 ignore_index: int = 8) -> None:
        super(FRNetPPT, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.losses = []
        closest_dir = find_closest_directory(log_path)
        if closest_dir is None:
            self.log_path = log_path + "/temp_log"
        else:
            self.log_path = log_path + "/" + closest_dir      
        print("Recording Losses to: ", self.log_path)

        self.ignore_index = ignore_index
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.save_output = save_output

    def freeze_feature_extractor(self):
        # freeze all parameters
        self.freeze(self.voxel_encoder)
        self.freeze(self.backbone)
        self.freeze(self.decode_head)
        self.freeze(self.auxiliary_head)
        self.freeze(self.data_preprocessor)

        # unfreeze non feature extractor elements
        self.unfreeze(self.data_preprocessor.embedding)
        self.unfreeze(self.decode_head.final_classifier)
    
    def freeze(self, module):
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
    
    def unfreeze(self, module):
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from points."""
        voxel_dict = self.voxel_encoder(batch_inputs_dict['voxels'])
        voxel_dict = self.backbone(voxel_dict)
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict

    def save_losses(self):
        file_path = self.log_path + "/" + "losses.csv"
        os.makedirs(self.log_path, exist_ok=True)

        # Check if the file exists to decide whether to append to it
        mode = 'a' if os.path.exists(file_path) else 'w'
        
        with open(file_path, mode) as f:
            losses = np.array(self.losses)
            losses = losses.mean(axis=0)
            loss_sum = np.sum(losses)
            f.write(str(loss_sum) + ",")
            for i, item in enumerate(losses):
                f.write(str(item))
                if i == len(losses) - 1:
                    f.write('\n')
                else:
                    f.write(',')
        
        self.losses = []
    
    def save_pcd_clouds(self, batch_inputs_dict, results):
        points = [points.detach().cpu().numpy() for points in batch_inputs_dict['points']]
        ambient = [ambient.detach().cpu().numpy() for ambient in batch_inputs_dict['ambient']]
        pred_labels = [result.pred_pts_seg.get('pts_semantic_mask').cpu().numpy() for result in results]
        gt_labels = [result.eval_ann_info.get('pts_semantic_mask', None) for result in results]
        file_paths = [file_path for file_path in batch_inputs_dict['file_path']]
        
        assert len(set([len(points), len(ambient), len(pred_labels), len(gt_labels)])) == 1, "Must be same length"

        for i, pts in enumerate(points):
            cloud = np.hstack((points[i], ambient[i].reshape(-1, 1)))
            gt_label = gt_labels[i] if gt_labels[i] is not None else np.zeros_like(pred_labels[i])
            labels = np.hstack((pred_labels[i].reshape(-1, 1), gt_label.reshape(-1, 1)))
            
            # remove range interpolated values and set correct types
            cloud = cloud[:labels.shape[0], :].astype(np.float32)
            labels = labels.astype(np.uint32)
            
            # generate a save filepath
            original_path = file_paths[i]
            parts = original_path.split('/')
            seq = parts[-3]
            cloud_name = parts[-1].replace('.bin', '')
            save_path = self.log_path + "/output_clouds/" + seq + "/" + cloud_name + ".pcd"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # save pcd to the file path
            fields = ('x', 'y', 'z', 'intensity', 'ambient', 'pred_label', 'gt_label')
            types = (np.float32, np.float32, np.float32, np.float32, np.float32, np.uint32, np.uint32)
            points = [cloud[:,0], cloud[:,1], cloud[:,2], cloud[:,3], cloud[:,4], labels[:,0], labels[:,1]]

            pc = PointCloud.from_points(points, fields, types)
            pc.save(save_path)
        

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        # perform check to ensure most of the cloud is valid
        for sample in batch_data_samples:
            point_count = sample.gt_pts_seg.pts_semantic_mask.shape[0]
            invalid_count = (sample.gt_pts_seg.pts_semantic_mask == self.ignore_index).detach().sum().item()
            invalid_ratio = invalid_count / point_count
            if invalid_ratio > 0.5:
                print("WARNING: invalid ratio higher than 50% (", invalid_ratio, "), skipping frame...")
                return dict()

        # extract features using backbone
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)

        self.losses.append([val.detach().item() for val in losses.values()])
        return losses

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        batch_input_metas = []

        if len(self.losses) > 0:
            self.save_losses()

        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict,
                                                   batch_input_metas,
                                                   self.test_cfg)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)

        results = self.postprocess_result(seg_logits_list, batch_data_samples)
        if self.save_output:
            self.save_pcd_clouds(batch_inputs_dict, results)
        return results

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> dict:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: Forward output of model without any post-processes.
        """
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)
