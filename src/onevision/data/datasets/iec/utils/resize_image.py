#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2

from onevision.core import progress_bar
from onevision.core import VisionBackend
from onevision.imgproc import resize
from onevision.io import create_dirs
from onevision.io import is_image_file
from onevision.io import read_image
from onevision.utils import datasets_dir

iec22_dir = os.path.join(datasets_dir, "iec", "iec22")
splits    = ["train"]

with progress_bar() as pbar:
	for split in splits:
		pattern = os.path.join(iec22_dir, split, "*", "*")
		for image_path in pbar.track(glob.glob(pattern)):
			if not is_image_file(image_path):
				continue
			image   = read_image(image_path, VisionBackend.CV)
			image   = resize(image, [512, 512, 3])
			image   = image[:, :, ::-1]  # BGR -> RGB
			path    = image_path.replace(f"iec22", "iec22_512")
			new_dir = Path(path).parent
			create_dirs([new_dir])
			cv2.imwrite(path, image)
