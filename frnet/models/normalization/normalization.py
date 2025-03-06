from typing import Union, Tuple, List

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from torch import Tensor

@MODELS.register_module('PPTPointNorm')
class PPTPointNorm(nn.Module):
    """Point Prompt Training inspired normalization layer.

    Args:
        num_features (int): :math:`C` from an expected input of size
            :math:`(N, C)`
    """

    def __init__(self,
                 num_features: int,
                 base_norm: nn.Module,
                 context_channels: int = 256,
                 apply_ppt: bool = True) -> None:
        super(PPTPointNorm, self).__init__()
        self.mlp = nn.Sequential(
             nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
        )
        self.base_norm = base_norm
        self.apply_ppt = apply_ppt

    def forward(self, x: Tensor, dataset_token: Tensor, coors: Tensor) -> Tensor:
        assert x.shape[0] > 0, 'PPTPointNorm does not support empty inputs'

        # compute norm and apply ppt on top
        output = self.base_norm(x)

        if (self.apply_ppt):
            shift, scale = self.mlp(dataset_token).chunk(2, dim=1)

            for i in range(int(coors[-1, 0].item() + 1)):
                mask = (coors[:, 0]==i)
                output[mask, :] = output[mask, :] * (1.0 + scale[i, :]) + shift[i,:]
            return output
        else:
            return output

@MODELS.register_module('PPTNorm')
class PPTNorm(nn.Module):
    """Point Prompt Training inspired normalization layer.

    Args:
        num_features (int): :math:`C` from an expected input of size
            :math:`(N, C, +)`
    """

    def __init__(self,
                 num_features: int,
                 base_norm: nn.Module,
                 context_channels: int = 256,
                 apply_ppt: bool = True) -> None:
        super(PPTNorm, self).__init__()
        self.mlp = nn.Sequential(
             nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
        )
        self.base_norm = base_norm
        self.apply_ppt = apply_ppt

    def forward(self, x: Tensor, dataset_token: Tensor) -> Tensor:
        assert x.shape[0] > 0, 'PPTNorm does not support empty inputs'

        # compute norm and apply ppt on top
        intermediate = self.base_norm(x)

        if (self.apply_ppt):
            shift, scale = self.mlp(dataset_token).chunk(2, dim=1)
            
            # get dimensions to properly match
            extra_dims = intermediate.dim() - shift.dim()
            for _ in range(extra_dims):
                shift = shift.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            
            output = intermediate * (1.0 + scale) + shift
            return output
        else:
            return intermediate

@MODELS.register_module('SequentialPassthrough')
class SequentialPassthrough(nn.Module):
    """Sequence of modules with variable number of inputs and outputs (expects num input to equal num outputs)

    Args:
        modules (dict {int: nn.Module}): a dictionary with the key being the number of inputs and outputs,
        and the value being the module.
    """

    def __init__(self, modules: List[Tuple[int, nn.Module]]) -> None:
        super(SequentialPassthrough, self).__init__()
        self.io_counts, self.module_list = zip(*modules) 
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x: Tensor, dataset_tokens: Tensor, coors = None) -> Tensor:
        for idx, module in enumerate(self.module_list):
            num_io = self.io_counts[idx]
            if num_io is 1:
                x = module(x)
            elif num_io is 2:
                x = module(x, dataset_tokens)
            elif num_io is 3:
                if coors is None:
                    raise RuntimeError("Must provide coors")
                x = module(x, dataset_tokens, coors)
            else:
                raise RuntimeError("SequentialPassthrough only supports 1 or 2 input modules.")
        return x

def build_norm_layer_ppt(cfg: dict,
                     num_features: int,
                     context_channels: int = 256,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'base_norm' not in cfg or 'type' not in cfg["base_norm"]:
        raise KeyError('the cfg dict must contain a base_norm with the key "type"')
    if 'use_ppt' not in cfg:
        raise KeyError('the cfg dict must contain the key "use_ppt"')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    
    # compute base normalization layer
    name, base_layer = build_norm_layer(cfg["base_norm"].copy(), num_features, postfix)

    # now apply ppt normalization on top of base norm
    name = 'ppt_' + name
    if (cfg["type"] == "point"):
        layer = PPTPointNorm(num_features, base_layer, context_channels = context_channels, apply_ppt = cfg["use_ppt"])
    elif (cfg["type"] == "batch"):
        layer = PPTNorm(num_features, base_layer, context_channels = context_channels, apply_ppt = cfg["use_ppt"])
    else:
        raise RuntimeError("Unsupported PPT type: " + cfg["type"]) 
    return name, layer