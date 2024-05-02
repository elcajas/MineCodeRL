from __future__ import annotations
import hashlib
import logging
from omegaconf import OmegaConf
from typing import Callable, Literal
from mineclip import MineCLIP
from mineclip.utils.torch_utils import get_activation

import numpy as np
import torch.nn as nn

from .inference import load_model

def set_MineCLIP(cfg):
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    assert (hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum), "broken ckpt"

    model = MineCLIP(**cfg)
    model.load_ckpt(ckpt.path, strict=True)
    logging.info("MineCLIP successfully loaded with checkpoint")
    return model

def set_gDINO(cfg):
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return model

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def build_mlp(
        input_dim,
            *,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int = None,
        num_layers: int = None,
        activation: str | Callable = "relu",
        weight_init: str | Callable = "orthogonal",
        bias_init="zeros",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        add_input_activation: bool | str | Callable = False,
        add_input_norm: bool = False,
        add_output_activation: bool | str | Callable = False,
        add_output_norm: bool = False,
    ) -> nn.Sequential:

    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    pass

# def set_gdino(cfg, device):
#     model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
#     return model