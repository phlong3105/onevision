#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import pickle as pkl

import cv2
import torch

from onevision.cv.core import to_channel_first
from onevision.cv.core import to_channel_last
from onevision.cv.imgproc import adjust_gamma
from onevision.cv.io import read_image_cv


def draw_rect(im, cords, color=None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    
    cords = cords[:, :4]
    cords = cords.reshape(-1,4)

    if not color:
        color = [255,255,255]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
        
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
        print(pt1, pt2)
    return im


img    = read_image_cv("../data/messi.jpeg")
img    = img[:, :, ::-1]
bboxes = pkl.load(open("../data/messi_ann.pkl", "rb"))
bboxes = bboxes[:, 0:4]

img    = to_channel_first(img)
ten    = torch.from_numpy(img)

img    = adjust_gamma(img, 0.5)
ten    = adjust_gamma(ten, 0.5)
# ten    = torch.bitwise_not(ten)
bboxes = torch.from_numpy(bboxes)
img    = to_channel_last(img)
ten    = ten.numpy()
ten    = to_channel_last(ten)

# img = affine(img, angle=20, translate=[50, 50], scale=1.0, shear=[0, 0])
# ten = affine(ten, angle=20, translate=[50, 50], scale=1.0, shear=[0, 0])
"""
ten, bboxes = rotate_image_box(ten, bboxes, 45)

img    = np.stack([img, img])
print(img.shape)
img    = rgb_to_hsv(img)
img    = to_channel_last(img)
ten    = ten.numpy()
ten    = to_channel_last(ten)
bboxes = bboxes.numpy()
"""
# cv2.imshow("ten", draw_rect(ten, bboxes))
cv2.imshow("img", img)
cv2.imshow("ten", ten)
cv2.waitKey(0)
