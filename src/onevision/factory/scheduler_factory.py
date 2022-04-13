#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from typing import Union

from munch import Munch
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from onevision.factory.factory import Registry
from onevision.type import is_list_of
from onevision.utils import error_console

__all__ = [
    "SchedulerFactory"
]


class SchedulerFactory(Registry):
	"""Factory class for creating schedulers."""
	
	# MARK: Build
	
	def build(
		self, optimizer: Optimizer, name: Optional[str], *args, **kwargs
	) -> Optional[_LRScheduler]:
		"""Factory command to create a scheduler. This method gets the
		appropriate scheduler class from the registry and creates an detection
		of it, while passing in the parameters given in `kwargs`.
		
		Args:
			optimizer (Optimizer):
				Optimizer.
			name (str, optional):
				Scheduler's name.
		
		Returns:
			detection (_LRScheduler, optional):
				An detection of the scheduler that is created.
		"""
		if name is None:
			return None
		
		if name not in self.registry:
			error_console.log(f"{name} does not exist in the registry.")
			return None
		
		if name in ["gradual_warmup_scheduler"]:
			after_scheduler = kwargs.pop("after_scheduler")
			if isinstance(after_scheduler, dict):
				name_ = after_scheduler.pop("name")
				if name_ in self.registry:
					after_scheduler = self.registry[name_](
						optimizer=optimizer, **after_scheduler
					)
				else:
					after_scheduler = None
			return self.registry[name](
				optimizer=optimizer, after_scheduler=after_scheduler,
				*args, **kwargs
			)
		
		return self.registry[name](optimizer=optimizer, *args, **kwargs)
	
	def build_from_dict(
		self,
		optimizer: Optimizer,
		cfg      : Optional[Union[dict, Munch]],
		*args, **kwargs
	) -> Optional[_LRScheduler]:
		"""Factory command to create a scheduler. This method gets the
		appropriate scheduler class from the registry and creates an
		detection of it, while passing in the parameters given in `cfg`.

		Args:
			optimizer (Optimizer):
				Optimizer.
			cfg (dict, Munch, optional):
				Scheduler' config.

		Returns:
			detection (_LRScheduler, optional):
				An detection of the scheduler that is created.
		"""
		if cfg is None:
			return None
		
		if not isinstance(cfg, (dict, Munch)):
			raise TypeError(f"`cfg` must be a `dict`. But got: {type(cfg)}.")
		
		if "name" not in cfg:
			raise KeyError("`cfg` dict must contain the key `name`.")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(optimizer=optimizer, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		optimizer: Optimizer,
		cfgs     : Optional[list[Union[dict, Munch]]],
		*args, **kwargs
	) -> Optional[list[_LRScheduler]]:
		"""Factory command to create schedulers. This method gets the
		appropriate schedulers classes from the registry and creates
		detections of them, while passing in the parameters given in `cfgs`.

		Args:
			optimizer (Optimizer):
				Optimizer.
			cfgs (list[dict, Munch], optional):
				List of optimizers' configs.

		Returns:
			detection (list[Optimizer], optional):
				Instances of the scheduler that are created.
		"""
		if cfgs is None:
			return None
		
		if (
            not is_list_of(cfgs, expected_type=dict) or
            not is_list_of(cfgs, expected_type=Munch)
        ):
			raise TypeError(f"`cfgs` must be a `list[dict]`. But got: {type(cfgs)}.")
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(optimizer=optimizer, name=name, **cfg))
		
		return instances if len(instances) > 0 else None
	
	def build_from_list(
		self,
		optimizers: list[Optimizer],
		cfgs      : Optional[list[list[Union[dict, Munch]]]],
		*args, **kwargs
	) -> Optional[list[_LRScheduler]]:
		"""Factory command to create schedulers. This method gets the
		appropriate schedulers classes from the registry and creates
		detections of them, while passing in the parameters given in `cfgs`.

		Args:
			optimizers (list[Optimizer]):
				List of optimizers.
			cfgs (list[list[dict, Munch]], optional):
				2D-list of optimizers' configs.

		Returns:
			detection (list[Optimizer], optional):
				Instances of the scheduler that are created.
		"""
		if cfgs is None:
			return None
		
		if (
			not is_list_of(cfgs, expected_type=list) or
			not all(is_list_of(cfg, expected_type=dict) for cfg in cfgs)
        ):
			raise TypeError(f"`cfgs` must be a 2D `list[dict]`. But got: {type(cfgs)}.")
		
		if len(optimizers) != len(cfgs):
			raise ValueError(f"`optimizers` and `cfgs` must have the same length."
			                 f" But got: {len(optimizers)} != {len(cfgs)}.")
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for optimizer, cfgs in zip(optimizers, cfgs_):
			for cfg in cfgs:
				name  = cfg.pop("name")
				cfg  |= kwargs
				instances.append(
					self.build(optimizer=optimizer, name=name, **cfg)
				)
		
		return instances if len(instances) > 0 else None
