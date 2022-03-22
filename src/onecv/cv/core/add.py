#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from multipledispatch import dispatch
from torch import Tensor

from onecv.cv.core.channels import is_channel_first
from onecv.cv.core.channels import to_channel_first
from onecv.cv.core.channels import to_channel_last
from onecv.factory import TRANSFORMS
from onecv.type import TensorOrArray

__all__ = [
    "add_weighted",
    "blend_images",
    "make_image_grid",
    "AddWeighted",
]


# MARK: - Functional

def add_weighted(
    src1 : TensorOrArray,
    alpha: float,
    src2 : TensorOrArray,
    beta : float,
    gamma: float = 0.0,
) -> Tensor:
    """Calculate the weighted sum of two Tensors. Function calculates the
    weighted sum of two Tensors as follows:
        out = src1 * alpha + src2 * beta + gamma

    Args:
        src1 (TensorOrArray[B, C, H, W]):
            First image.
        alpha (float):
            Weight of the src1 elements.
        src2 (TensorOrArray[B, C, H, W]):
            Tensor of same size and channel number as src1 [*, H, W].
        beta (float):
            Weight of the src2 elements.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.

    Returns:
        add (Tensor[B, C, H, W]):
            Weighted tensor.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
        >>> output.shape
        torch.Size([1, 1, 5, 5])
    """
    if not isinstance(src1, Tensor):
        raise TypeError(f"`src1` must be a `Tensor`. But got: {type(src1)}.")
    if not isinstance(src2, Tensor):
        raise TypeError(f"`src2` must be a `Tensor`. But got: {type(src2)}.")
    if src1.shape != src2.shape:
        raise ValueError(f"`src1` and `src2` must have the same shape. "
                         f"But got: {src1.shape} != {src2.shape}.")
    if not isinstance(alpha, float):
        raise TypeError(f"`alpha` must be a `float`. But got: {type(alpha)}.")
    if not isinstance(beta, float):
        raise TypeError(f"`beta` must be a `float`. But got: {type(beta)}.")
    if not isinstance(gamma, float):
        raise TypeError(f"`gamma` must be a `float`. But got: {type(gamma)}.")

    return src1 * alpha + src2 * beta + gamma


@dispatch(Tensor, Tensor, float, float)
def blend_images(
    overlays: Tensor,
    images  : Tensor,
    alpha   : float,
    gamma   : float = 0.0
) -> Tensor:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (Tensor[B, C, H, W]):
            Images we want to overlay on top of the original image.
        images (Tensor[B, C, H, W]):
            Source images.
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):
            Default: `0.0`.

    Returns:
        blend (Tensor[B, C, H, W]):
            Blended image.
    """
    overlays_np = overlays.numpy()
    images_np   = images.numpy()
    blends      = blend_images(overlays_np, images_np, alpha, gamma)
    blends      = torch.from_numpy(blends)
    return blends


@dispatch(np.ndarray, np.ndarray, float, float)
def blend_images(
    overlays: np.ndarray,
    images  : np.ndarray,
    alpha   : float,
    gamma   : float = 0.0
) -> np.ndarray:
    """Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

    Args:
        overlays (np.ndarray[B, C, H, W]):
            Images we want to overlay on top of the original image.
        images (np.ndarray[B, C, H, W]):
            Source images.
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):
            Default: `0.0`.

    Returns:
        blend (np.ndarray[B, C, H, W]):
            Blended image.
    """
    # NOTE: Type checking
    if overlays.ndim != images.ndim:
        raise ValueError(f"`overlays` and `images` must have the same ndim. "
                         f"But got: {overlays.ndim} != {images.ndim}")
    
    # NOTE: Convert to channel-first
    overlays = to_channel_first(overlays)
    images   = to_channel_first(images)
    
    # NOTE: Unnormalize images
    from .normalize import denormalize_naive
    images = denormalize_naive(images)
    
    # NOTE: Convert overlays to same data type as images
    images   = images.astype(np.uint8)
    overlays = overlays.astype(np.uint8)
    
    # NOTE: If the images are of shape [CHW]
    if overlays.ndim == 3 and images.ndim == 3:
        return cv2.addWeighted(overlays, alpha, images, 1.0 - alpha, gamma)
    
    # NOTE: If the images are of shape [BCHW]
    if overlays.ndim == 4 and images.ndim == 4:
        if overlays.shape[0] != images.shape[0]:
            raise ValueError(
                f"`overlays` and `images` must have the same batch sizes. "
                f"But got: {overlays.shape[0]} != {images.shape[0]}"
            )
        blends = []
        for overlay, image in zip(overlays, images):
            blends.append(cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, gamma))
        blends = np.stack(blends, axis=0).astype(np.uint8)
        return blends


@dispatch(Tensor, int)
def make_image_grid(images: Tensor, nrow: int = 1) -> Tensor:
    """Concatenate multiple images into a single image.

    Args:
        images (Tensor):
            Images can be:
                - A 4D mini-batch image of shape [B, C, H, W].
                - A 3D RGB image of shape [C, H, W].
                - A 2D grayscale image of shape [H, W].
        nrow (int):
            Number of images in each row of the grid. Final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (Tensor):
            Concatenated image.
    """
    return torchvision.utils.make_grid(tensor=images, nrow=nrow)


@dispatch(np.ndarray, int)
def make_image_grid(images: np.ndarray, nrow: int = 1) -> np.ndarray:
    """Concatenate multiple images into a single image.

    Args:
        images (np.array):
            Images can be:
                - A 4D mini-batch image of shape [B, C, H, W].
                - A 3D RGB image of shape [C, H, W].
                - A 2D grayscale image of shape [H, W].
        nrow (int):
            Number of images in each row of the grid. Final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (np.ndarray):
            Concatenated image.
    """
    # NOTE: Type checking
    if images.ndim == 3:
        return images
    
    # NOTE: Conversion (just for sure)
    if is_channel_first(images):
        images = to_channel_last(images)
    
    b, c, h, w = images.shape
    ncols      = nrow
    nrows      = (b // nrow) if (b // nrow) > 0 else 1
    cat_image  = np.zeros((c, int(h * nrows), w * ncols))
    for idx, im in enumerate(images):
        j = idx // ncols
        i = idx % ncols
        cat_image[:, j * h: j * h + h, i * w: i * w + w] = im
    return cat_image


@dispatch(list, int)
def make_image_grid(images: list, nrow: int = 1) -> TensorOrArray:
    """Concatenate multiple images into a single image.

    Args:
        images (list):
            A list of images of the same shape [C, H, W].
        nrow (int):
            Number of images in each row of the grid. Final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (Image):
            Concatenated image.
    """
    if (isinstance(images, list) and
        all(isinstance(t, np.ndarray) for t in images)):
        cat_image = np.concatenate([images], axis=0)
        return make_image_grid(cat_image, nrow)
    elif isinstance(images, list) and all(torch.is_tensor(t) for t in images):
        return torchvision.utils.make_grid(tensor=images, nrow=nrow)
    else:
        raise TypeError(f"Do not support {type(images)}.")


@dispatch(dict, int)
def make_image_grid(images: dict, nrow: int = 1) -> TensorOrArray:
    """Concatenate multiple images into a single image.

    Args:
        images (dict):
            A dict of images of the same shape [C, H, W].
        nrow (int, optional):
            Number of images in each row of the grid. Final grid size is
            `[B / nrow, nrow]`. Default: `1`.

    Returns:
        cat_image (Image):
            Concatenated image.
    """
    if (isinstance(images, dict) and
        all(isinstance(t, np.ndarray) for k, t in images.items())):
        cat_image = np.concatenate(
            [image for key, image in images.items()], axis=0
        )
        return make_image_grid(cat_image, nrow)
    elif (isinstance(images, dict) and
          all(torch.is_tensor(t) for k, t in images.items())):
        values = list(tuple(images.values()))
        return torchvision.utils.make_grid(values, nrow)
    else:
        raise TypeError(f"Do not support {type(images)}.")


# MARK: - Modules

@TRANSFORMS.register(name="add_weighted")
class AddWeighted(nn.Module):
    """Calculate the weighted sum of two Tensors. Function calculates the
    weighted sum of two Tensors as follows:
        out = src1 * alpha + src2 * beta + gamma

    Args:
        alpha (float):
            Weight of the src1 elements.
        beta (float):
            Weight of the src2 elements.
        gamma (float):
            Scalar added to each sum.

    Example:
        >>> input1 = torch.rand(1, 1, 5, 5)
        >>> input2 = torch.rand(1, 1, 5, 5)
        >>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
        >>> output.shape
        torch.Size([1, 1, 5, 5])
    """
    
    # MARK: Magic Functions
    
    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    # MARK: Forward Pass
    
    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)
