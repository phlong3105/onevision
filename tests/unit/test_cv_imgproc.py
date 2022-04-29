#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test CV Image Processing operations.
"""

import numpy as np

from onevision import cosine_distance
from onevision import euclidean_distance


def test_cosine_distance():
	x = np.array((1, 2, 3))
	y = np.array((1, 1, 1))
	print(cosine_distance(x, y))


def test_euclidean_distance():
	x = np.array((1, 2, 3))
	y = np.array((1, 1, 1))
	print(euclidean_distance(x, y))
