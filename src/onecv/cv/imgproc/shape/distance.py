#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from math import asin
from math import cos
from math import pow
from math import sin
from math import sqrt

import numpy as np
from multipledispatch import dispatch

from onecv.factory import DISTANCES

__all__ = [
	"angle_between_vectors",
	"chebyshev_distance",
	"cosine_distance",
	"euclidean_distance",
	"hausdorff_distance",
	"haversine_distance",
	"manhattan_distance",
	"ChebyshevDistance",
	"CosineDistance",
	"EuclideanDistance",
	"HausdorffDistance",
	"HaversineDistance",
	"ManhattanDistance",
]


# MARK: - Functional

def angle_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculate angle of 2 trajectories between two trajectories.
	"""
	vec1 = np.array([x[-1][0] - x[0][0],
	                 x[-1][1] - x[0][1]])
	vec2 = np.array([y[-1][0] - y[0][0],
	                 y[-1][1] - y[0][1]])
	
	L1 = np.sqrt(vec1.dot(vec1))
	L2 = np.sqrt(vec2.dot(vec2))
	
	if L1 == 0 or L2 == 0:
		return False
	
	cos   = vec1.dot(vec2) / (L1 * L2)
	angle = np.arccos(cos) * 360 / (2 * np.pi)
	
	return angle


@dispatch(np.ndarray, np.ndarray)
def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Chebyshev distance is a metric defined on a vector space where the
	distance between two vectors is the greatest of their differences along any
	coordinate dimension.
	"""
	n   = x.shape[0]
	ret = -1 * np.inf
	for i in range(n):
		d = abs(x[i] - y[i])
		if d > ret:
			ret = d
	return ret


@dispatch(np.ndarray, np.ndarray)
def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Cosine similarity is a measure of similarity between two non-zero
	vectors of an inner product space.
	"""
	n 	= x.shape[0]
	xy_dot = 0.0
	x_norm = 0.0
	y_norm = 0.0
	for i in range(n):
		xy_dot += x[i] * y[i]
		x_norm += x[i] * x[i]
		y_norm += y[i] * y[i]
	return 1.0 - xy_dot / (sqrt(x_norm) * sqrt(y_norm))


@dispatch(np.ndarray, np.ndarray)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Euclidean distance btw 2 arrays."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += ((x[i] - y[i]) ** 2)
	return sqrt(ret)


def hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Hausdorff distance btw 2 arrays.
	
	`euclidean_distance`, `manhattan_distance`, `chebyshev_distance`,
	`cosine_distance`, `haversine_distance` could be use for this function.
	"""
	cmax = 0.0
	for i in range(len(x)):
		cmin = np.inf
		for j in range(len(y)):
			d = euclidean_distance(x[i, :], y[j, :])
			if d < cmin:
				cmin = d
			if cmin < cmax:
				break
		if cmax < cmin < np.inf:
			cmax = cmin
	return cmax


@dispatch(np.ndarray, np.ndarray)
def haversine_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Haversine (or great circle) distance is the angular distance between two
	points on the surface of a sphere. First coordinate of each point is assumed
	to be the latitude, the second is the longitude, given in radians.
	Dimension of the data must be 2.
	"""
	R 		= 6378.0
	radians = np.pi / 180.0
	lat_x 	= radians * x[0]
	lon_x 	= radians * x[1]
	lat_y 	= radians * y[0]
	lon_y 	= radians * y[1]
	dlon  	= lon_y - lon_x
	dlat  	= lat_y - lat_x
	a 		= (pow(sin(dlat / 2.0), 2.0) +
	            cos(lat_x) * cos(lat_y) * pow(sin(dlon / 2.0), 2.0))
	return R * 2 * asin(sqrt(a))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
	"""Calculation of Manhattan distance btw 2 arrays."""
	n   = x.shape[0]
	ret = 0.0
	for i in range(n):
		ret += abs(x[i] - y[i])
	return ret


# MARK: - Modules

@DISTANCES.register(name="chebyshev")
class ChebyshevDistance:
	"""Calculate Chebyshev distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return chebyshev_distance(x=x, y=y)


@DISTANCES.register(name="cosine")
class CosineDistance:
	"""Calculate Cosine distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return cosine_distance(x=x, y=y)


@DISTANCES.register(name="euclidean")
class EuclideanDistance:
	"""Calculate Euclidean distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return euclidean_distance(x=x, y=y)


@DISTANCES.register(name="hausdorff")
class HausdorffDistance:
	"""Calculate Hausdorff distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return hausdorff_distance(x=x, y=y)


@DISTANCES.register(name="haversine")
class HaversineDistance:
	"""Calculate Haversine distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return haversine_distance(x=x, y=y)


@DISTANCES.register(name="manhattan")
class ManhattanDistance:
	"""Calculate Manhattan distance."""

	# MARK: Magic Functions

	def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
		return manhattan_distance(x=x, y=y)
