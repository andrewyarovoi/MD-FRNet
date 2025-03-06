import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

def cross_entropy_floating(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none')

    if weight is not None:
        loss = loss * weight

    eps = torch.finfo(torch.float32).eps
    if ignore_index is not None and ignore_index != -100:
        mask = (label[:, ignore_index] == 0.0).float()
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + eps)
    else:
        loss = loss.mean()

    return loss

@MODELS.register_module()
class CrossEntropyLossExtended(CrossEntropyLoss):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLossExtended, self).__init__(
            use_sigmoid, use_mask, reduction, class_weight, ignore_index, loss_weight, avg_non_ignore)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        # if label is float one hots, use custom cross entropy function
        if label.dim() == 2 and (label.dtype == torch.float32 or label.dtype == torch.float64):
            reduction = (reduction_override if reduction_override else self.reduction)
            assert reduction is 'mean', "Only mean is supported for float one hots"
            assert avg_factor is None, "Only none average factor is supported for float one hots"
            if ignore_index is None:
                ignore_index = self.ignore_index

            if self.class_weight is not None:
                class_weight = cls_score.new_tensor(
                    self.class_weight, device=cls_score.device)
            else:
                class_weight = None
            
            loss_cls = self.loss_weight * cross_entropy_floating(
                cls_score,
                label,
                weight,
                class_weight=class_weight,
                ignore_index=ignore_index,
                avg_non_ignore=self.avg_non_ignore,
                **kwargs)

            return loss_cls
                
        # if label is not float one hots, use default behavior
        return super().forward(
            cls_score,
            label,
            weight,
            avg_factor,
            reduction_override,
            ignore_index,
            **kwargs
        )