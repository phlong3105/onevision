#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from enum import Enum

import cv2

__all__ = [
    "InterpolationMode",
    "interpolation_mode_from_int",
    "cv_modes_mapping",
    "pil_modes_mapping",
]


# MARK: - Enum

class InterpolationMode(Enum):
    """Interpolation modes. Available interpolation methods are:
     [`nearest`, `bilinear`, `bicubic`, `box`, `hamming`, and `lanczos`].
    """
    BICUBIC       = "bicubic"
    BILINEAR      = "bilinear"
    NEAREST       = "nearest"
    # For PIL compatibility
    BOX           = "box"
    HAMMING       = "hamming"
    LANCZOS       = "lanczos"
    # For opencv compatibility
    AREA          = "area"
    CUBIC         = "cubic"
    LANCZOS4      = "lanczos4"
    LINEAR        = "linear"
    LINEAR_EXACT  = "linear_exact"
    MAX           = "max"
    NEAREST_EXACT = "nearest_exact"


def interpolation_mode_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0 : InterpolationMode.NEAREST,
        1 : InterpolationMode.LANCZOS,
        2 : InterpolationMode.BILINEAR,
        3 : InterpolationMode.BICUBIC,
        4 : InterpolationMode.BOX,
        5 : InterpolationMode.HAMMING,
        6 : InterpolationMode.AREA,
        7 : InterpolationMode.CUBIC,
        8 : InterpolationMode.LANCZOS4,
        9 : InterpolationMode.LINEAR,
        10: InterpolationMode.MAX,
    }
    return inverse_modes_mapping[i]


cv_modes_mapping = {
    InterpolationMode.AREA    : cv2.INTER_AREA,
    InterpolationMode.CUBIC   : cv2.INTER_CUBIC,
    InterpolationMode.LANCZOS4: cv2.INTER_LANCZOS4,
    InterpolationMode.LINEAR  : cv2.INTER_LINEAR,
    InterpolationMode.MAX     : cv2.INTER_MAX,
    InterpolationMode.NEAREST : cv2.INTER_NEAREST,
}


pil_modes_mapping = {
    InterpolationMode.NEAREST : 0,
    InterpolationMode.LANCZOS : 1,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC : 3,
    InterpolationMode.BOX     : 4,
    InterpolationMode.HAMMING : 5,
}
