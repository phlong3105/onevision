# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Union

import numpy as np
import PIL.Image
import torch
from PIL import ExifTags
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional_pil import _is_pil_image

from onevision.cv.core.channels import is_channel_first
from onevision.math import make_divisible
from onevision.type import Int2Or3T
from onevision.type import Int2T
from onevision.type import TensorOrArray
from onevision.utils import error_console

__all__ = [
    "check_image_size",
    "get_exif_size",
    "get_image_center",
    "get_image_center4",
    "get_image_hw",
    "get_image_size",
]


# MARK: - Functional

# Get orientation exif tag

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def check_image_size(image_size: Int2Or3T, stride: int = 32) -> int:
    """Verify image size is a multiple of stride and return the new size.
    
    Args:
        image_size (Int2Or3T):
            Image size.
        stride (int):
            Stride. Default: `32`.
    
    Returns:
        new_size (int):
            Appropriate size.
    """
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 3:  # [H, W, C]
            image_size = image_size[1]
        elif len(image_size) == 2:  # [H, W]
            image_size = image_size[0]
        
    new_size = make_divisible(image_size, int(stride))  # ceil gs-multiple
    if new_size != image_size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (image_size, stride, new_size)
        )
    return new_size


def get_exif_size(image: Image) -> Int2T:
    """Return the exif-corrected PIL size."""
    size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = (size[1], size[0])
        elif rotation == 8:  # rotation 90
            size = (size[1], size[0])
    except:
        pass
    return size[1], size[0]


def get_image_center(image: TensorOrArray) -> TensorOrArray:
    """Get image center as (x=h/2, y=w/2).
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image.
   
    Returns:
        center (TensorOrArray):
            Image center as (x=h/2, y=w/2).
    """
    h, w   = get_image_hw(image)
    center = np.array((h / 2, w / 2))
    
    if isinstance(image, Tensor):
        return torch.from_numpy(center)
    elif isinstance(image, np.ndarray):
        return center
    else:
        TypeError(f"Unexpected type {type(image)}")


def get_image_center4(image: TensorOrArray) -> TensorOrArray:
    """Get image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image.
   
    Returns:
        center (TensorOrArray):
            Image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    """
    h, w   = get_image_hw(image)
    center = np.array((h / 2, w / 2))
    center = np.hstack((center, center))
    
    if isinstance(image, Tensor):
        return torch.from_numpy(center)
    elif isinstance(image, np.ndarray):
        return center
    else:
        TypeError(f"Unexpected type {type(image)}")


def get_image_hw(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int2T:
    """Returns the size of an image as [H, W].
    
    Args:
        image (Tensor, np.ndarray, PIL Image):
            The image to be checked.
   
    Returns:
        size (Int2T):
            Image size as [H, W].
    """
    if isinstance(image, (Tensor, np.ndarray)):
        if is_channel_first(image):  # [.., C, H, W]
            return [image.shape[-2], image.shape[-1]]
        else:  # [.., H, W, C]
            return [image.shape[-3], image.shape[-2]]
    elif _is_pil_image(image):
        return list(image.size)
    else:
        TypeError(f"Unexpected type {type(image)}")


# MARK: - Alias

get_image_size = get_image_hw
