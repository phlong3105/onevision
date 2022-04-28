#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for modeling the motion of an individual tracked object.
"""

from __future__ import annotations

import abc

from onevision.data.data_class import Detection

__all__ = [
    "Motion",
]


# MARK: - Modules

class Motion(metaclass=abc.ABCMeta):
    """Motion implements the base template to model how an individual tracked
    object moves. It is used for predicting the next position of the tracked
    object.

    Attributes:
        hits (int):
            Number of frame has that track appear.
        hit_streak (int):
            Number of `consecutive` frame has that track appear.
        age (int):
            Number of frame while the track is alive, from
            Candidate -> Deleted.
        time_since_update (int):
            Number of `consecutive` frame that track disappears.
        history (list):
            Store all the `predict` position of track in z-bounding box value,
            these position appear while no bounding matches the track if any
            bounding box matches the track, then history = [].
    """

    # MARK: Magic Functions

    def __init__(
        self,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
        *args, **kwargs
    ):
        self.hits              = hits
        self.hit_streak        = hit_streak
        self.age               = age
        self.time_since_update = time_since_update
        self.history           = []

    # MARK: Update

    @abc.abstractmethod
    def update(self, detection: Detection, *args, **kwargs):
        """Updates the state of the motion model with observed features.

		Args:
			detection (Detection):
				Get the specific features used to update the motion model from
				new detection of the object.
		"""
        pass

    @abc.abstractmethod
    def predict(self):
        """Advances the state of the motion model and returns the predicted
        estimate.
        """
        pass

    @abc.abstractmethod
    def current(self):
        """Returns the current motion model estimate."""
        pass
