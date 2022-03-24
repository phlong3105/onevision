#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from typing import Union

import numpy as np
import torch.nn.functional as F
from torch import Tensor

from onevision.cv.core.image import get_image_size
from onevision.cv.core.image import is_channel_first
from onevision.type import FloatAnyT
from onevision.type import Int2Or3T
from onevision.type import TensorOrArray
from onevision.type import to_size

__all__ = [
    "PaddingMode",
    "padding_mode_from_int",
    "pad_image",
]


# MARK: - Enum

class PaddingMode(Enum):
    """Padding modes. Available padding methods are:
    """
    CONSTANT      = "constant"
    # For torch compatibility
    CIRCULAR      = "circular"
    REFLECT       = "reflect"
    REPLICATE     = "replicate"
    # For numpy compatibility
    EDGE          = "edge"
    EMPTY         = "empty"
    LINEAR_RAMP   = "linear_ramp"
    MAXIMUM       = "maximum"
    MEAN          = "mean"
    MEDIAN        = "median"
    MINIMUM       = "minimum"
    SYMMETRIC     = "symmetric"
    WRAP          = "wrap"

    @staticmethod
    def values() -> list:
        return [e.value for e in PaddingMode]
    
    
def padding_mode_from_int(i: int) -> PaddingMode:
    inverse_modes_mapping = {
        0 : PaddingMode.CONSTANT,
        1 : PaddingMode.CIRCULAR,
        2 : PaddingMode.REFLECT,
        3 : PaddingMode.REPLICATE,
        4 : PaddingMode.EDGE,
        5 : PaddingMode.EMPTY,
        6 : PaddingMode.LINEAR_RAMP,
        7 : PaddingMode.MAXIMUM,
        8 : PaddingMode.MEAN,
        9 : PaddingMode.MEDIAN,
        10: PaddingMode.MINIMUM,
        11: PaddingMode.SYMMETRIC,
        12: PaddingMode.WRAP,
    }
    return inverse_modes_mapping[i]


# MARK: - Functional

def pad_image(
    image   : TensorOrArray,
    pad_size: Int2Or3T,
    mode    : Union[PaddingMode, str] = "constant",
    value   : Optional[FloatAnyT]     = 0.0,
) -> TensorOrArray:
    """Pad image with `value`.
    
    Args:
        image (TensorOrArray[B, C, H, W]/[B, H, W, C]):
            Image to be padded.
        pad_size (Int2Or3T[H, W, *]):
            Padded image size.
        mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        value (FloatAnyT, optional):
            Fill value for `constant` padding. Default: `0.0`.
            
    Returns:
        image (TensorOrArray[B, C, H, W]/[B, H, W, C]):
            Padded image.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"`image.ndim` must be 3 or 4. "
                         f"But got: {image.ndim}")
    if isinstance(mode, str) and mode not in PaddingMode.values():
        raise ValueError(f"`mode` must be one of: {PaddingMode.values()}. "
                         f"But got {mode}.")
    elif isinstance(mode, PaddingMode):
        if mode not in PaddingMode:
            raise ValueError(f"`mode` must be one of: {PaddingMode}. "
                             f"But got: {mode}.")
        mode = mode.value
    if isinstance(image, Tensor):
        if mode not in ("constant", "circular", "reflect", "replicate"):
            raise ValueError()
    if isinstance(image, np.ndarray):
        if mode not in ("constant", "edge", "empty", "linear_ramp", "maximum",
                        "mean", "median", "minimum", "symmetric", "wrap"):
            raise ValueError()
    
    h0, w0 = get_image_size(image)
    h1, w1 = to_size(pad_size)
    # Image size > pad size, do nothing
    if (h0 * w0) >= (h1 * w1):
        return image
    
    if value is None:
        value = 0
    pad_h = int(abs(h0 - h1) / 2)
    pad_w = int(abs(w0 - w1) / 2)

    if isinstance(image, Tensor):
        if is_channel_first(image):
            pad = (pad_w, pad_w, pad_h, pad_h)
        else:
            pad = (0, 0, pad_w, pad_w, pad_h, pad_h)
        return F.pad(input=image, pad=pad, mode=mode, value=value)
    elif isinstance(image, np.ndarray):
        if is_channel_first(image):
            if image.ndim == 3:
                pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
            else:
                pad_width = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
        else:
            if image.ndim == 3:
                pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            else:
                pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0))
        return np.pad(array=image, pad_width=pad_width, mode=mode, constant_values=value)
    
    return image
