from __future__ import annotations
import hashlib
import logging
from omegaconf import OmegaConf
from typing import Callable, Literal
from mineclip import MineCLIP
from mineclip.utils.torch_utils import get_initializer

import numpy as np
import torch.nn as nn
import loralib as lora
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from peft import LoraConfig, get_peft_model

from .inference import load_model, load_model_with_lora

def print_trainable_parameters(model):
    r"""Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def set_MineCLIP(cfg):
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    assert (hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum), "broken ckpt"

    model = MineCLIP(**cfg)
    model.load_ckpt(ckpt.path, strict=True)
    logging.info("MineCLIP successfully loaded with checkpoint")
    return model

def set_gDINO(cfg, device):
    if cfg.agent.train_image_model:
        model = load_model_with_lora("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth", device, rank=8, lora_alpha=2)
        lora.mark_only_lora_as_trainable(model)
        print_trainable_parameters(model)
    
    else:
        model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth", device, train=cfg.agent.train_image_model)
    return model

def set_hf_gDINO(cfg, device):
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    config = LoraConfig(
    r = 8,
    lora_alpha=16,
    target_modules = [
        "backbone.conv_encoder.model.encoder.layers.0.blocks.0.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.0.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.0.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.0.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.0.output.dense",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.1.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.1.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.1.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.1.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.0.blocks.1.output.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.0.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.0.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.0.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.0.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.0.output.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.1.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.1.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.1.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.1.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.1.blocks.1.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.0.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.0.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.0.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.0.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.0.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.1.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.1.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.1.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.1.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.1.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.2.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.2.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.2.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.2.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.2.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.3.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.3.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.3.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.3.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.3.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.4.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.4.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.4.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.4.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.4.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.5.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.5.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.5.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.5.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.2.blocks.5.output.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.0.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.0.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.0.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.0.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.0.output.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.1.attention.self.query",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.1.attention.self.value",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.1.attention.output.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.1.intermediate.dense",
        "backbone.conv_encoder.model.encoder.layers.3.blocks.1.output.dense",
        "vision_proj",
        "text_proj",
        "values_vision_proj",
        "values_text_proj",
        "out_vision_proj",
        "out_text_proj",
        "self_attn.sampling_offsets",
        "self_attn.attention_weights",
        "self_attn.value_proj",
        "self_attn.output_proj",
        "deformable_layer.fc1",
        "deformable_layer.fc2",
    ],
    lora_dropout=0.1,
    bias='none',
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
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
    """
    In other popular RL implementations, tanh is typically used with orthogonal
    initialization, which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    act_layer = get_activation(activation)

    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)

    if norm_type is not None:
        norm_type = norm_type.lower()

    if not norm_type:
        norm_type = nn.Identity
    elif norm_type == "batchnorm":
        norm_type = nn.BatchNorm1d
    elif norm_type == "layernorm":
        norm_type = nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")

    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), norm_type(hidden_dim), act_layer()]
        for i in range(hidden_depth - 1):
            mods += [
                nn.Linear(hidden_dim, hidden_dim),
                norm_type(hidden_dim),
                act_layer(),
            ]
        mods.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        mods = [norm_type(input_dim)] + mods
    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        mods = [act_layer()] + mods
    if add_output_norm:
        mods.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        mods.append(act_layer())

    for mod in mods:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*mods)

def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    elif callable(activation):
        return activation
    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": lambda: nn.ReLU(),
        "leaky_relu": lambda: nn.LeakyReLU(inplace=True),
        "swish": lambda: nn.SiLU(inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]