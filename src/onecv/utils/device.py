#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from enum import Enum
from time import time
from typing import Any
from typing import Optional

import torch
from pynvml import *
from torch import Tensor

__all__ = [
    "extract_device_dtype",
    "get_gpu_memory",
    "memory_unit_conversion_from_b",
    "memory_unit_from_int",
    "memory_unit_from_str",
    "MemoryUnit",
    "select_device",
    "select_device_old",
    "time_synchronized",
]


# MARK: - Memory Unit Mode

class MemoryUnit(Enum):
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"


def memory_unit_from_str(s: str) -> MemoryUnit:
    s = s.lower()
    inverse_modes_mapping = {
        "b"  : MemoryUnit.B,
        "kb" : MemoryUnit.KB,
        "mb" : MemoryUnit.MB,
        "gb" : MemoryUnit.GB,
        "tb" : MemoryUnit.TB,
        "pb" : MemoryUnit.PB,
    }
    return inverse_modes_mapping[s]


def memory_unit_from_int(i: int) -> MemoryUnit:
    inverse_modes_mapping = {
        0: MemoryUnit.B,
        1: MemoryUnit.KB,
        2: MemoryUnit.MB,
        3: MemoryUnit.GB,
        4: MemoryUnit.TB,
        5: MemoryUnit.PB,
    }
    return inverse_modes_mapping[i]


memory_unit_conversion_from_b = {
    MemoryUnit.B : 1024 ** 0,
    MemoryUnit.KB: 1024 ** 1,
    MemoryUnit.MB: 1024 ** 2,
    MemoryUnit.GB: 1024 ** 3,
    MemoryUnit.TB: 1024 ** 4,
    MemoryUnit.PB: 1024 ** 5,
}


# MARK: - Functional

def extract_device_dtype(
    tensor_list: list[Optional[Any]]
) -> tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are
    Tensor). If so, it would return a tuple of (device, dtype).
    Default: (cpu, `get_default_dtype()`).

    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (Tensor,)):
                continue
            _device = tensor.device
            _dtype  = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype  = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError(
                    f"Passed values must be in the same `device` and `dtype`. "
                    f"But got: ({device}, {dtype}) and ({_device}, {_dtype})."
                )
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device("cpu")
    if dtype is None:
        dtype  = torch.get_default_dtype()
    return device, dtype


def get_gpu_memory(
    device_index: int = 0, unit: Union[MemoryUnit, str, int] = MemoryUnit.GB
) -> tuple[int, int, int]:
    if isinstance(unit, str):
        unit = memory_unit_from_str(unit)
    elif isinstance(unit, int):
        unit = memory_unit_from_int(unit)
    if unit not in MemoryUnit:
        from ..utils import error_console
        error_console.log(f"Unknown memory unit: {unit}")
        unit = MemoryUnit.GB
        
    nvmlInit()
    h     = nvmlDeviceGetHandleByIndex(device_index)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = memory_unit_conversion_from_b[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return total, used, free


def select_device(
    model_name: str           = "",
    device    : Optional[str] = "",
    batch_size: Optional[int] = None
) -> torch.device:
    """Select the device to runners the model.
    
    Args:
        model_name (str):
            Name of the model.
        device (str, optional):
            Name of device for running. Can be: 'cpu' or '0' or '0,1,2,3'.
        batch_size (int, optional):
            Number of samples in one forward & backward pass.

    Returns:
        device (torch.device):
            GPUs or CPU.
    """
    from ..utils import console
    
    if isinstance(device, int):
        device = f"{device}"
        
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # If device requested other than `cpu`
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        if not torch.cuda.is_available():  # Check availability
            raise ValueError(f"CUDA unavailable, invalid device {device} requested")

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        bytes_to_mb = 1024 ** 2  # bytes to MB
        num_gpus    = torch.cuda.device_count()
        if num_gpus > 1 and batch_size:  # check that batch_size is compatible with device_count
            if batch_size % num_gpus != 0:
                raise ValueError(f"batch-size {batch_size} not multiple of GPU count {num_gpus}")
        x = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]
        s = "Using CUDA "
        for i in range(0, num_gpus):
            if i == 1:
                s = " " * len(s)
            console.log(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / bytes_to_mb)
            )
    else:
        console.log("Using CPU")

    console.log("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def select_device_old(
    model_name: str           = "",
    device    : Optional[str] = "",
    batch_size: Optional[int] = None
) -> torch.device:
    """Select the device to runners the model.
    
    Args:
        model_name (str):
            Name of the model.
        device (str, optional):
            Name of device for running.
        batch_size (int, optional):
            Number of samples in one forward & backward pass.

    Returns:
        device (torch.device):
            GPUs or CPU.
    """
    from ..utils import console
    
    if device is None:
        return torch.device("cpu")

    # device = 'cpu' or '0' or '0,1,2,3'
    s   = f"{model_name}"  # string
    
    if isinstance(device, str) and device.lower() == "cpu":
        cpu = True
    else:
        cpu = False
    
    if cpu:
        # Force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        # Non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # Check availability
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA unavailable, invalid device {device} requested.")
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        n = torch.cuda.device_count()

        # Check that batch_size is compatible with device_count
        if n > 1 and batch_size:
            if batch_size % n != 0:
                raise ValueError(f"batch-size {batch_size} not multiple of GPU count {n}.")
        space = " " * len(s)
        
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"
    
    console.log(s)
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
