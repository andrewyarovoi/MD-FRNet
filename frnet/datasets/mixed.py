# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np
import pickle

from mmdet3d.registry import DATASETS
from torch.utils.data import Dataset
from mmdet3d.datasets import Seg3DDataset

LABEL2CAT = {
    0: 'ground', 
    1: 'road', 
    2: 'vegetation', 
    3: 'structure', 
    4: 'vehicle', 
    5: 'humans', 
    6: 'object', 
    7: 'outliers'
}

@DATASETS.register_module()
class MixedDataset():
    """Mixed Dataset.

    This class serves as a dataset of datasets, loading inputs from the appropriate dataset 
    based on given context and index.

    Args:
        datasets (dict): Dictionary of context names and associated datasets
    """

    def __init__(self,
                 datasets: Dict[str, dict] = {None}) -> None:
        self.datasets = {key: DATASETS.build(dataset) for key, dataset in datasets.items()}
        self.cloud_filepaths = {}
        for key, dataset_config in datasets.items():
            file_path = dataset_config['data_root'] + dataset_config['ann_file']
            with open(file_path, 'rb') as file:
                temp = pickle.load(file)
                file_paths = [val['lidar_points']['lidar_path'] for val in temp['data_list']]
                self.cloud_filepaths[key] = file_paths

        self.dataset_lengths_cumsum = np.cumsum([len(dataset) for dataset in self.datasets.values()])
        self.dataset_offsets = [0] + list(self.dataset_lengths_cumsum)
        self.len = self.dataset_lengths_cumsum[-1]
        self.metainfo = {"label2cat": LABEL2CAT}
        print("LENGTH of datasets: ", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.dataset_lengths_cumsum, idx, side="right")
        offset_idx = idx - self.dataset_offsets[dataset_idx]
        key = list(self.datasets.keys())[dataset_idx]
        sample = self.datasets[key][offset_idx]
        
        sample['file_path'] = self.cloud_filepaths[key][offset_idx]
        sample['context'] = key
        return sample