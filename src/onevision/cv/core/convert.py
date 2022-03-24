# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import deepcopy
from typing import Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.transforms.functional import _is_numpy
from torchvision.transforms.functional_pil import _is_pil_image

from onevision.cv.core.channels import get_num_channels
from onevision.cv.core.channels import to_channel_first
from onevision.cv.core.channels import to_channel_last
from onevision.factory import TRANSFORMS
from onevision.type import TensorOrArray
from onevision.utils import error_console

__all__ = [
    "is_integer_image",
    "is_one_hot_image",
    "to_image",
    "to_pil_image",
    "to_tensor",
    "ToImage",
    "ToTensor",
]


# MARK: - Functional

def is_integer_image(image: TensorOrArray) -> bool:
    """Check if the given image is integer-encoded."""
    c = get_num_channels(image)
    if c == 1:
        return True
    return False


def is_one_hot_image(image: TensorOrArray) -> bool:
    """Check if the given image is one-hot encoded."""
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


def to_image(tensor: Tensor, keep_dim: bool = True, denormalize: bool = False) -> np.ndarray:
    """Converts a PyTorch tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        tensor (Tensor):
            Image of the form [H, W], [C, H, W] or [B, H, W, C].
        keep_dim (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
        
    Returns:
        image (np.ndarray):
            Image of the form [H, W], [H, W, C] or [B, H, W, C].
    """
    if not torch.is_tensor(tensor):
        error_console.log(f"Input type is not a Tensor. Got: {type(tensor)}.")
        return tensor
    if tensor.ndim > 4 or tensor.ndim < 2:
        raise ValueError(f"`tensor.ndim` must be == 2, 3, 4, or 5. But got: {tensor.ndim}.")

    image = tensor.cpu().detach().numpy()
    
    # NOTE: Channel last format
    image = to_channel_last(image, keep_dim=keep_dim)
    
    # NOTE: Denormalize
    if denormalize:
        from .normalize import denormalize_naive
        image = denormalize_naive(image)
        
    return image.astype(np.uint8)


def to_pil_image(image: TensorOrArray) -> PIL.Image:
    """Convert image from `np.ndarray` or `Tensor` to PIL image."""
    if torch.is_tensor(image):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return torchvision.transforms.ToPILImage()(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
    image    : Union[np.ndarray, PIL.Image],
    keep_dim : bool = True,
    normalize: bool = False,
) -> Tensor:
    """Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.
    
    Args:
        image (np.ndarray, PIL.Image):
            Image in [H, W, C], [H, W] or [B, H, W, C].
        keep_dim (bool):
            If `False` unsqueeze the image to match the shape [B, H, W, C].
            Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    
    Returns:
        img (Tensor):
            Converted image.
    """
    if not (_is_pil_image(image) or _is_numpy(image) or torch.is_tensor(image)):
        raise TypeError(f"`image` must be `PIL.Image`, `np.ndarray`, or `Tensor`. "
                        f"But got: {type(image)}.")
    
    if ((_is_numpy(image) or torch.is_tensor(image))
        and (image.ndim > 4 or image.ndim < 2)):
        raise ValueError(f"`image.ndim` must be == 2, 3, or 4. But got: {image.ndim}.")

    # img = image
    img = deepcopy(image)
    
    # NOTE: Handle PIL Image
    if _is_pil_image(img):
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = np.array(img, mode_to_nptype.get(img.mode, np.uint8), copy=True)
        if image.mode == "1":
            img = 255 * img
    
    # NOTE: Handle numpy array
    if _is_numpy(img):
        img = torch.from_numpy(img).contiguous()
    
    # NOTE: Channel first format
    img = to_channel_first(img, keep_dim=keep_dim)
   
    # NOTE: Normalize
    if normalize:
        from .normalize import normalize_naive
        img = normalize_naive(img)
    
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=torch.get_default_dtype())
    return img


# MARK: - Modules

@TRANSFORMS.register(name="to_image")
class ToImage(nn.Module):
    """Converts a PyTorch tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        keep_dim (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
    """

    def __init__(self, keep_dim: bool = True, denormalize: bool = False):
        super().__init__()
        self.keep_dim    = keep_dim
        self.denormalize = denormalize

    def forward(self, image: Tensor) -> np.ndarray:
        return to_image(image, self.keep_dim, self.denormalize)


@TRANSFORMS.register(name="to_tensor")
class ToTensor(nn.Module):
    """Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.

    Args:
        keep_dim (bool):
            If `False` unsqueeze the image to match the shape [B, H, W, C].
            Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    """

    def __init__(self, keep_dim: bool = False, normalize: bool = False):
        super().__init__()
        self.keep_dim  = keep_dim
        self.normalize = normalize

    def forward(self, image: Union[np.ndarray, PIL.Image]) -> Tensor:
        return to_tensor(image, self.keep_dim, self.normalize)
