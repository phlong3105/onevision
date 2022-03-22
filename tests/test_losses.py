#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test Loss Functions
"""

from __future__ import annotations

import cv2

from onecv import loss
from onecv import print_table
from onecv import resize
from onecv import to_tensor

image = cv2.imread("image.jpg")
image = to_tensor(image)
image = resize(image, size=[256, 256, 3])
image = image.unsqueeze(0)

gt    = cv2.imread("gt.jpg")
gt    = to_tensor(gt)
gt    = resize(gt, size=[256, 256, 3])
gt    = gt.unsqueeze(0)

# NOTE: Loss Functions
loss_functions = [
	loss.CharbonnierEdgeLoss(eps=1e-3, loss_weight=[1.0, 0.05], reduction="mean"),
	loss.CharbonnierLoss(eps=1e-3, loss_weight=1.0, reduction="mean"),
	loss.ColorConstancyLoss(loss_weight=1.0, reduction="mean"),
	loss.EdgeLoss(eps=1e-3, loss_weight=1.0, reduction="mean"),
	loss.ExposureControlLoss(patch_size=16, mean_val=0.6, loss_weight=1.0, reduction="mean"),
	loss.GradientLoss(loss_weight=1.0, reduction="mean"),
	loss.GrayLoss(loss_weight=1.0, reduction="mean"),
	loss.GrayscaleLoss(loss_weight=1.0, reduction="mean"),
	loss.IlluminationSmoothnessLoss(tv_loss_weight=1, loss_weight=1.0, reduction="mean"),
	loss.MAELoss(loss_weight=1.0, reduction="mean"),
	loss.MSELoss(loss_weight=1.0, reduction="mean"),
	loss.NonBlurryLoss(loss_weight=1.0, reduction="mean"),
	loss.PSNRLoss(max_val=1.0, loss_weight=1.0, reduction="mean"),
	loss.SmoothMAELoss(beta=1.0, loss_weight=1.0, reduction="mean"),
	loss.SpatialConsistencyLoss(loss_weight=1.0, reduction="mean"),
	loss.SSIMLoss(window_size=3, max_val=1.0, eps=1e-12, loss_weight=1.0, reduction="mean"),
]

losses = {}
for loss_func in loss_functions:
	losses[loss_func.name] = loss_func(input=image, pred=image, target=gt)

print_table(losses)
