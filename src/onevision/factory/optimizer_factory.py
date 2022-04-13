#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from typing import Union

import torch
from munch import Munch
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from onevision.factory.factory import Registry
from onevision.type import is_list_of
from onevision.utils import error_console

__all__ = [
    "OptimizerFactory"
]


class OptimizerFactory(Registry):
	"""Factory class for creating optimizers."""
	
	# MARK: Build
	
	def build(
		self, net: torch.nn.Module, name: str, *args, **kwargs
	) -> Optional[Optimizer]:
		"""Factory command to create an optimizer. This method gets the
		appropriate optimizer class from the registry and creates an detection
		of it, while passing in the parameters given in `kwargs`.
		
		Args:
			net (nn.Module):
				Neural network module.
			name (str):
				Optimizer's name.
		
		Returns:
			detection (Optimizer, optional):
				An detection of the optimizer that is created.
		"""
		if name not in self.registry:
			error_console.log(f"{name} does not exist in the registry.")
			return None
		
		return self.registry[name](params=net.parameters(), *args, **kwargs)
	
	def build_from_dict(
		self, net: torch.nn.Module, cfg: Optional[Union[dict, Munch]], **kwargs
	) -> Optional[Optimizer]:
		"""Factory command to create an optimizer. This method gets the
		appropriate optimizer class from the registry and creates an detection
		of it, while passing in the parameters given in `cfg`.

		Args:
			net (nn.Module):
				Neural network module.
			cfg (dict, Munch, optional):
				Optimizer' config.

		Returns:
			detection (Optimizer, optional):
				An detection of the optimizer that is created.
		"""
		if cfg is None:
			return None
		
		if not isinstance(cfg, (dict, Munch)):
			error_console.log("`cfg` must be a dict.")
			return None
		
		if "name" not in cfg:
			error_console.log("`cfg` dict must contain the key `name`.")
			return None
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(net=net, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		net : torch.nn.Module,
		cfgs: Optional[list[Union[dict, Munch]]],
		**kwargs
	) -> Optional[list[Optimizer]]:
		"""Factory command to create optimizers. This method gets the
		appropriate optimizers classes from the registry and creates
		detections of them, while passing in the parameters given in `cfgs`.

		Args:
			net (nn.Module):
				List of neural network modules.
			cfgs (list[dict, Munch], optional):
				List of optimizers' configs.

		Returns:
			detection (list[Optimizer], optional):
				Instances of the optimizers that are created.
		"""
		if cfgs is None:
			return None
		
		if (not is_list_of(cfgs, expected_type=dict) 
			or not is_list_of(cfgs, expected_type=Munch)):
			error_console.log("`cfgs` must be a list of dict.")
			return None
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(net=net, name=name, **cfg))
		
		return instances if len(instances) > 0 else None
	
	def build_from_list(
		self,
		nets: list[torch.nn.Module],
		cfgs: Optional[list[Union[dict, Munch]]],
		*args, **kwargs
	) -> Optional[list[Optimizer]]:
		"""Factory command to create optimizers. This method gets the
		appropriate optimizers classes from the registry and creates
		detections of them, while passing in the parameters given in `cfgs`.

		Args:
			nets (list[nn.Module]):
				List of neural network modules.
			cfgs (list[dict, Munch]):
				List of optimizers' configs.

		Returns:
			detection (list[Optimizer], optional):
				Instances of the optimizers that are created.
		"""
		if cfgs is None:
			return None
		
		if (not is_list_of(cfgs, expected_type=dict) 
			or not is_list_of(cfgs, expected_type=Munch)):
			raise TypeError(f"`cfgs` must be a `list` of `dict`. But got: {cfgs}.")
		
		if not is_list_of(nets, expected_type=dict):
			raise TypeError(f"`nets` must be a `list[nn.Module]`. But got: {nets}")
		
		if len(nets) != len(cfgs):
			raise ValueError(f"`nets` and `cfgs` must have the same length. "
			                 f" But got: {len(nets)} != {len(cfgs)}.")
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for net, cfg in zip(nets, cfgs_):
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(net=net, name=name, **cfg))
		
		return instances if len(instances) > 0 else None
