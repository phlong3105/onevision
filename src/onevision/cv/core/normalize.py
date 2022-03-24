# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing functions for intensity normalisation.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from multipledispatch import dispatch
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.transforms.functional import normalize

from onevision.factory import TRANSFORMS
from onevision.type import TensorOrArray

__all__ = [
    "denormalize",
    "denormalize_naive",
    "is_normalized",
    "normalize",
    "normalize_min_max",
    "normalize_naive",
    "Denormalize",
    "Normalize",
]


# MARK: - Functional

def denormalize(data: Tensor, mean: Union[Tensor, float], std: Union[Tensor, float]) -> Tensor:
    """Denormalize an image/video image with mean and standard deviation.
    
    input[channel] = (input[channel] * std[channel]) + mean[channel]
        
        where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n`
        channels,

    Args:
        data (Tensor[B, C, *, *]):
            Image.
        mean (Tensor[B, C, *, *], float):
            Mean for each channel.
        std (Tensor[B, C, *, *], float):
            Standard deviations for each channel.

    Return:
        out (Tensor[B, N, *, *]):
            Denormalized image with same size as input.

    Examples:
        >>> x   = torch.rand(1, 4, 3, 3)
        >>> out = denormalize(x, 0.0, 255.)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x    = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std  = 255. * torch.ones(1, 4)
        >>> out  = denormalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """
    shape = data.shape

    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=data.device,
                            dtype=data.dtype)
    if isinstance(std, float):
        std  = torch.tensor([std] * shape[1], device=data.device,
                            dtype=data.dtype)
    if not isinstance(data, Tensor):
        raise TypeError(f"`data` should be a `Tensor`. But got: {type(data)}")
    if not isinstance(mean, Tensor):
        raise TypeError(f"`mean` should be a `Tensor`. But got: {type(mean)}")
    if not isinstance(std, Tensor):
        raise TypeError(f"`std` should be a `Tensor`. But got: {type(std)}")

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"`mean` and `data` must have the same shape. "
                             f"But got: {mean.shape} and {data.shape}.")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"`std` and `data` must have the same shape. "
                             f"But got: {std.shape} and {data.shape}.")

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std  = torch.as_tensor(std,  device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std  = std[..., :, None]

    out = (data.view(shape[0], shape[1], -1) * std) + mean
    return out.view(shape)


@dispatch((Tensor, np.ndarray))
def denormalize_naive(image: TensorOrArray) -> TensorOrArray:
    if isinstance(image, Tensor):
        return torch.clamp(image * 255, 0, 255).to(torch.uint8)
    elif isinstance(image, np.ndarray):
        return np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Do not support: {type(image)}.")
    

@dispatch(list)
def denormalize_naive(image: list) -> list:
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        return list(denormalize_naive(np.array(image)))
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        return [denormalize_naive(i) for i in image]
    
    # NOTE: List of Tensor
    if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
        return list(denormalize_naive(torch.stack(image)))
    if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
        return [denormalize_naive(i) for i in image]
    
    raise TypeError(f"Do not support {type(image)}.")


@dispatch(tuple)
def denormalize_naive(image: tuple) -> tuple:
    image = list(image)
    image = denormalize_naive(image)
    return tuple(image)


@dispatch(dict)
def denormalize_naive(image: dict) -> dict:
    if not all(isinstance(v, (tuple, list, Tensor, np.ndarray))
               for k, v in image.items()):
        raise ValueError()
    
    for k, v in image.items():
        image[k] = denormalize_naive(v)
    
    return image


def is_normalized(image: TensorOrArray) -> TensorOrArray:
    if isinstance(image, Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"Do not support: {type(image)}.")
    

def normalize_min_max(
    image  : Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    eps    : float = 1e-6
) -> Tensor:
    """Normalise an image/video image by MinMax and re-scales the value
    between a range.

    Args:
        image (Tensor[B, C, *, *]):
            Image to be normalized.
        min_val (float):
            Minimum value for the new range.
        max_val (float):
            Maximum value for the new range.
        eps (float):
            Float number to avoid zero division.

    Returns:
        x_out (Tensor[B, C, *, *]):
            Fnormalized tensor image with same shape.

    Example:
        >>> x      = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(image, min_val=-1., max_val=1.)
        >>> x_norm.min()
        image(-1.)
        >>> x_norm.max()
        image(1.0000)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"data should be a image. But got: {type(image)}.")
    if not isinstance(min_val, float):
        raise TypeError(f"`min_val` should be a `float`. But got: {type(min_val)}.")
    if not isinstance(max_val, float):
        raise TypeError(f"`max_val` should be a `float`. But got: {type(max_val)}.")
    if image.ndim < 3:
        raise ValueError(f"`image.ndim` must be >= 3. But got: {image.shape}.")

    shape = image.shape
    B, C  = shape[0], shape[1]

    x_min = image.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max = image.view(B, C, -1).max(-1)[0].view(B, C, 1)

    x_out = ((max_val - min_val) * (image.view(B, C, -1) - x_min) /
             (x_max - x_min + eps) + min_val)
    return x_out.view(shape)


@dispatch((Tensor, np.ndarray))
def normalize_naive(image: TensorOrArray) -> TensorOrArray:
    """Convert image from `torch.uint8` type and range [0, 255] to `torch.float`
    type and range of [0.0, 1.0].
    """
    if isinstance(image, Tensor):
        if abs(torch.max(image)) > 1.0:
            return image.to(torch.get_default_dtype()).div(255.0)
        else:
            return image.to(torch.get_default_dtype())
    elif isinstance(image, np.ndarray):
        if abs(np.amax(image)) > 1.0:
            return image.astype(np.float32) / 255.0
        else:
            return image.astype(np.float32)
    else:
        raise TypeError(f"Do not support: {type(image)}.")
    

@dispatch(list)
def normalize_naive(image: list) -> list:
    # NOTE: List of np.ndarray
    if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
        image = normalize_naive(np.array(image))
        return list(image)
    if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
        image = [normalize_naive(i) for i in image]
        return image
    
    # NOTE: List of Tensor
    if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
        image = normalize_naive(torch.stack(image))
        return list(image)
    if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
        image = [normalize_naive(i) for i in image]
        return image

    raise TypeError(f"Do not support {type(image)}.")


@dispatch(tuple)
def normalize_naive(image: tuple) -> tuple:
    image = list(image)
    image = normalize_naive(image)
    return tuple(image)


@dispatch(dict)
def normalize_naive(image: dict) -> dict:
    if not all(isinstance(v, (tuple, list, Tensor, np.ndarray))
               for k, v in image.items()):
        raise ValueError()
    
    for k, v in image.items():
        image[k] = normalize_naive(v)
    
    return image


# MARK: - Modules

@TRANSFORMS.register(name="denormalize")
class Denormalize(nn.Module):
    """Denormalize a tensor image with mean and standard deviation.
 
    Args:
        mean (Tensor[B, C, *, *], float):
            Mean for each channel.
        std (Tensor[B, C, *, *], float):
            Standard deviations for each channel.

    Examples:
        >>> x   = torch.rand(1, 4, 3, 3)
        >>> out = Denormalize(0.0, 255.)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3])

        >>> x    = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std  = 255. * torch.ones(1, 4)
        >>> out  = Denormalize(mean, std)(x)
        >>> out.shape
        torch.Size([1, 4, 3, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        super().__init__()
        self.mean = mean
        self.std  = std

    def __repr__(self):
        repr = f"(mean={self.mean}, std={self.std})"
        return self.__class__.__name__ + repr

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return denormalize(image, self.mean, self.std)
