#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""`cv` package has the same organization and structure as `cv2` library.
"""

from __future__ import annotations

from .core import *
from .core import image
from .imgproc import *
from .imgproc.shape import box
from .io import *
from .patching import *
from .stitching import *
from .utils import *
from .video import *
