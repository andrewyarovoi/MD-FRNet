# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS
from .get_closest_dir import find_closest_directory

from tabulate import tabulate
import os

@METRICS.register_module()
class MultiSegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 ignore_index: int,
                 log_path: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.ignore_index = ignore_index
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.log_path = log_path
        self.metric_counter = 0

        closest_dir = find_closest_directory(log_path)
        if closest_dir is None:
            self.log_path = log_path + "/temp_log"
        else:
            self.log_path = log_path + "/" + closest_dir      
        print("Recording Metrics to: ", self.log_path)

        super(MultiSegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        context = data_batch["context"]
        for idx, data_sample in enumerate(data_samples):
            pred_3d = data_sample['pred_pts_seg']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((context[idx], eval_ann_info, cpu_pred_3d))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        print("WARN: MultiSegMetric does not currently support submission formatting!")
        # submission_prefix = self.submission_prefix
        # if submission_prefix is None:
        #     tmp_dir = tempfile.TemporaryDirectory()
        #     submission_prefix = osp.join(tmp_dir.name, 'results')
        # mmcv.mkdir_or_exist(submission_prefix)
        # print("INFO: Saving results to ", submission_prefix)
        # # need to map network output to original label idx
        # cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
        #     np.int64)
        # for original_label, output_idx in self.dataset_meta['label2cat'].items(
        # ):
        #     if output_idx != self.ignore_index:
        #         cat2label[output_idx] = original_label

        # for i, (eval_ann, result) in enumerate(results):
        #     sample_idx = eval_ann['point_cloud']['lidar_idx']
        #     pred_sem_mask = result['semantic_mask'].numpy().astype(np.int64)
        #     pred_label = cat2label[pred_sem_mask]
        #     curr_file = f'{submission_prefix}/{sample_idx}.txt'
        #     np.savetxt(curr_file, pred_label, fmt='%d')

    def save_metrics(self, metrics):
        file_path = self.log_path + "/" + "metrics.txt"
        os.makedirs(self.log_path, exist_ok=True)

        # Check if the file exists to decide whether to append to it
        mode = 'a' if os.path.exists(file_path) else 'w'
        
        with open(file_path, mode) as f:
            f.write("---------------\n")
            f.write("Evaluation: " + str(self.metric_counter) + "\n")
            f.write("---------------\n\n")

            for context, data in metrics.items():
                f.write("Metrics for context " + context + ":\n")
                headers = data.keys()
                rows = [list(data.values())]
                table = tabulate(rows, headers=headers, tablefmt="grid")
                f.write(table + "\n\n")

                vals = list(data.values())
                print(context, ": ", [np.round(val, 4) for val in vals[-3:] + vals[:-3]])
        
        self.metric_counter += 1   

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']

        contexts = []
        gt_semantic_masks = []
        pred_semantic_masks = []
        
        for context, eval_ann, single_pred_results in results:
            contexts.append(context)
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks.append(single_pred_results['pts_semantic_mask'])

        # split samples by context
        unique_contexts = set(contexts)
        split_gt_sm = {}
        split_pred_sm = {}
        for context in unique_contexts:
            split_gt_sm[context] = [gt_semantic_masks[i] for i, ctx in enumerate(contexts) if ctx == context]
            split_pred_sm[context] = [pred_semantic_masks[i] for i, ctx in enumerate(contexts) if ctx == context]

        all_results = {}

        # evaluate each context individually
        for context in unique_contexts:
            print("Results for ", context, ":")
            results = seg_eval(
                split_gt_sm[context],
                split_pred_sm[context],
                label2cat,
                self.ignore_index,
                logger=logger)
            all_results[context] = results.copy()
        
        print("Results for Combined Dataset:")

        # return combined data
        ret_dict = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            self.ignore_index,
            logger=logger)
        
        all_results["combined"] = ret_dict.copy()

        # save results to text file
        self.save_metrics(all_results)

        return ret_dict
