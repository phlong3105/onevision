# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import copy

import numpy as np
from multipledispatch import dispatch
from torch import Tensor

from onevision.type import TensorOrArray

__all__ = [
	"get_num_channels",
	"is_channel_first",
	"is_channel_last",
	"to_channel_first",
	"to_channel_last",
]


# MARK: - Functional

def get_num_channels(image: TensorOrArray) -> int:
    """Get number of channels of the image."""
    if image.ndim == 4:
        if is_channel_first(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
        return c
    elif image.ndim == 3:
        if is_channel_first(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
        return c
    else:
        raise ValueError(f"`image.ndim` must be == 3 or 4. But got: {image.ndim}.")
    
    
def is_channel_first(image: TensorOrArray) -> bool:
    """Check if the image is in channel first format."""
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    
    raise ValueError(f"`image.ndim` must be == 3, 4, or 5. But got: {image.ndim}.")


def is_channel_last(image: TensorOrArray) -> bool:
    """Check if the image is in channel last format."""
    return not is_channel_first(image)


@dispatch(Tensor, keep_dim=bool)
def to_channel_first(image: Tensor, keep_dim: bool = True) -> Tensor:
    """Convert image to channel first format.
    
    Args:
        image (Tensor):
            Image.
        keep_dim (bool):
            If `False` unsqueeze the image to match the shape [B, H, W, C].
            Default: `True`.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image    = image.unsqueeze(0)
    elif image.ndim == 3:
        image    = image.permute(2, 0, 1)
    elif image.ndim == 4:
        image    = image.permute(0, 3, 1, 2)
        keep_dim = True
    elif image.ndim == 5:
        image    = image.permute(0, 1, 4, 2, 3)
        keep_dim = True
    else:
        raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")

    return image.unsqueeze(0) if not keep_dim else image


@dispatch(np.ndarray, keep_dim=bool)
def to_channel_first(image: np.ndarray, keep_dim: bool = True) -> np.ndarray:
    """Convert image to channel first format.
    
    Args:
        image (np.ndarray):
            Image.
        keep_dim (bool):
            If `False` unsqueeze the image to match the shape [B, H, W, C].
            Default: `True`.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image    = np.expand_dims(image, 0)
    elif image.ndim == 3:
        image    = np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        image    = np.transpose(image, (0, 3, 1, 2))
        keep_dim = True
    elif image.ndim == 5:
        image    = np.transpose(image, (0, 1, 4, 2, 3))
        keep_dim = True
    else:
        raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")

    return np.expand_dims(image, 0) if not keep_dim else image


@dispatch(Tensor, keep_dim=bool)
def to_channel_last(image: Tensor, keep_dim: bool = True) -> Tensor:
    """Convert image to channel last format.
    
    Args:
        image (Tensor):
            Image.
        keep_dim (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Default: `True`.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = image.permute(1, 2, 0)
    elif image.ndim == 4:  # [B, C, H, W] -> [B, H, W, C]
        image = image.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keep_dim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = image.permute(0, 1, 3, 4, 2)
        if input_shape[0] == 1 and not keep_dim:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")
    
    return image
    

@dispatch(np.ndarray, keep_dim=bool)
def to_channel_last(image: np.ndarray, keep_dim: bool = True) -> np.ndarray:
    """Convert image to channel last format.
    
    Args:
        image (np.ndarray):
            Image.
        keep_dim (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Default: `True`.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        image = np.transpose(image, (0, 2, 3, 1))
        if input_shape[0] == 1 and not keep_dim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = np.transpose(image, (0, 1, 3, 4, 2))
        if input_shape[0] == 1 and not keep_dim:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")
   
    return image
