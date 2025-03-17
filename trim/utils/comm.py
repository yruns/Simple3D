# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed engine.
Modified from detectron2(https://github.com/facebookresearch/detectron2)
"""

import inspect
import os
import random
from typing import *

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True


def get_callable_arguments(func_or_class):
    signature = inspect.signature(func_or_class)
    arguments = [param.name for param in signature.parameters.values() if param.name != 'self']
    return arguments


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def str_parameters(model):
    return list([n for n, p in model.named_parameters() if p.requires_grad])


def print_parameters_with_learnable(model):
    ret = []
    for n, p in model.named_parameters():
        ret.append((n, p.requires_grad))

    return ret


def sum_model_parameters(model):
    total_params = 0.0
    for param in model.parameters():
        total_params += torch.sum(param)
    return total_params


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def convert_and_move_tensor(input_value, dtype, device):
    input_value = convert_tensor_to_dtype(input_value, dtype)
    input_value = move_tensor_to_device(input_value, device)
    return input_value


def to_device(input_value, device):
    if device is None:
        return input_value.cuda()
    return input_value.to(device)


def move_tensor_to_device(input_value, device=None):
    """Move input tensors to device"""
    if isinstance(input_value, torch.Tensor):
        return to_device(input_value, device)

    # convert tuple to list
    if isinstance(input_value, tuple):
        return tuple(move_tensor_to_device(list(input_value), device))

    if isinstance(input_value, list):
        for i in range(len(input_value)):
            input_value[i] = move_tensor_to_device(input_value[i], device)
        return input_value

    if isinstance(input_value, dict):
        for key in input_value.keys():
            input_value[key] = move_tensor_to_device(input_value[key], device)
        return input_value

    if isinstance(input_value, DataBase):
        for key in input_value.__dict__.keys():
            input_value.__dict__[key] = convert_tensor_to_dtype(input_value.__dict__[key], device)

    return input_value


STR_TO_DTYPE = {
    "no": torch.float32,  # Default dtype
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "half": torch.half,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class DataBase:
    pass


def convert_str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch.dtype"""
    if dtype_str not in STR_TO_DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return STR_TO_DTYPE[dtype_str]


def convert_tensor_to_dtype(input_value, dtype: Union[torch.dtype, str], ignore_keys=None):
    """Move input tensors to device"""
    if isinstance(dtype, str):
        if dtype == "no":
            return input_value
        dtype = convert_str_to_dtype(dtype)

    # Tensor
    if isinstance(input_value, torch.Tensor):
        if input_value.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
            return input_value
        return input_value.to(dtype)

    # Tuple
    if isinstance(input_value, tuple):
        return tuple(convert_tensor_to_dtype(list(input_value), dtype))

    # List
    if isinstance(input_value, list):
        for i in range(len(input_value)):
            input_value[i] = convert_tensor_to_dtype(input_value[i], dtype)
        return input_value

    # Dict
    if isinstance(input_value, dict):
        for key in input_value.keys():
            if ignore_keys is not None and key in ignore_keys:
                continue
            input_value[key] = convert_tensor_to_dtype(input_value[key], dtype)
        return input_value

    if isinstance(input_value, DataBase):
        for key in input_value.__dict__.keys():
            input_value.__dict__[key] = convert_tensor_to_dtype(input_value.__dict__[key], dtype)

    return input_value


def copy_codebase(output_dir, exclude_dirs=None):
    """Copy codebase to output_dir for future reference"""
    from .dist import is_main_process
    if not is_main_process():
        return

    import shutil

    codebase_path = os.getcwd()
    output_dir = os.path.join(output_dir, "codebase")

    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", "wandb", "data", "output", "checkpoints", "visualize", "pretrained"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in os.listdir(codebase_path):
        if item in exclude_dirs:
            continue
        s = os.path.join(codebase_path, item)
        d = os.path.join(output_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True,
                            ignore=shutil.ignore_patterns("*.pyc", "*.pth"), dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
