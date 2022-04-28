#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base factory class for creating and registering classes.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Optional
from typing import Union

import torch
from munch import Munch
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from onevision.core.collection import is_list_of
from onevision.core.type import Callable
from onevision.utils import console
from onevision.utils import error_console
from onevision.utils import print_table

__all__ = [
	"Factory",
	"OptimizerFactory",
	"Registry",
	"SchedulerFactory",
]


# MARK: - Modules

class Registry:
	"""Base registry class for registering classes.

	Attributes:
		name (str):
			Registry name.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, name: str):
		self._name     = name
		self._registry = {}
	
	def __len__(self):
		return len(self._registry)
	
	def __contains__(self, key: str):
		return self.get(key) is not None
	
	def __repr__(self):
		format_str = self.__class__.__name__ \
					 + f"(name={self._name}, items={self._registry})"
		return format_str
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""Return the registry's name."""
		return self._name
	
	@property
	def registry(self) -> dict:
		"""Return the registry's dictionary."""
		return self._registry
	
	def get(self, key: str) -> Callable:
		"""Get the registry record of the given `key`."""
		if key in self._registry:
			return self._registry[key]
	
	# MARK: Register
	
	def register(
		self,
		name  : Optional[str] = None,
		module: Callable	  = None,
		force : bool          = False
	) -> callable:
		"""Register a module.

		A record will be added to `self._registry`, whose key is the class name
		or the specified name, and value is the class itself. It can be used
		as a decorator or a normal function.

		Example:
			# >>> backbones = Factory("backbone")
			# >>>
			# >>> @backbones.register()
			# >>> class ResNet:
			# >>>     pass
			# >>>
			# >>> @backbones.register(name="mnet")
			# >>> class MobileNet:
			# >>>     pass
			# >>>
			# >>> class ResNet:
			# >>>     pass
			# >>> backbones.register(ResNet)

		Args:
			name (str, optional):
				Module name to be registered. If not specified, the class
				name will be used.
			module (type):
				Module class to be registered.
			force (bool):
				Whether to override an existing class with the same name.
		"""
		if not (name is None or isinstance(name, str)):
			raise TypeError(f"`name` must be `None` or `str`. "
			                f"But got: {type(name)}.")
		
		# NOTE: Use it as a normal method: x.register(module=SomeClass)
		if module is not None:
			self.register_module(module, name, force)
			return module
		
		# NOTE: Use it as a decorator: @x.register()
		def _register(cls):
			self.register_module(cls, name, force)
			return cls
		
		return _register
	
	def register_module(
		self,
		module_class: Callable,
		module_name : Optional[str] = None,
		force	    : bool 			= False
	):
		if not inspect.isclass(module_class):
			raise TypeError(f"`module_class` must be a class type. "
			                f"But got: {type(module_class)}.")
		
		if module_name is None:
			module_name = module_class.__name__.lower()
		
		if isinstance(module_name, str):
			module_name = [module_name]
		
		for name in module_name:
			if not force and name in self._registry:
				continue
				# logger.debug(f"{name} is already registered in {self.name}.")
			else:
				self._registry[name] = module_class
	
	# MARK: Print

	def print(self):
		"""Print the registry dictionary."""
		console.log(f"[red]{self.name}:")
		print_table(self.registry)


class Factory(Registry):
	"""Default factory class for creating objects.
	
	Registered object could be built from registry.
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
	"""
	
	# MARK: Build
	
	def build(self, name: str, *args, **kwargs) -> object:
		"""Factory command to create an detection of the class. This method gets
		the appropriate class from the registry and creates an detection of
		it, while passing in the parameters given in `kwargs`.
		
		Args:
			name (str):
				Name of the class to create.
			
		Returns:
			detection (object, optional):
				An detection of the class that is created.
		"""
		if name not in self.registry:
			error_console.log(f"{name} does not exist in the registry.")
			return None
		
		instance = self.registry[name](*args, **kwargs)
		if not hasattr(instance, "name"):
			instance.name = name
		return instance
	
	def build_from_dict(
		self, cfg: Optional[Union[dict, Munch]], **kwargs
	) -> Optional[object]:
		"""Factory command to create an detection of a class. This method gets
		the appropriate class from the registry while passing in the
		parameters given in `cfg`.
		
		Args:
			cfg (dict, Munch):
				Class object' config.
		
		Returns:
			detection (object, optional):
				An detection of the class that is created.
		"""
		if cfg is None:
			return None
		
		if not isinstance(cfg, (dict, Munch)):
			error_console.log("`cfg` must be a dict.")
			return None
		
		if "name" not in cfg:
			error_console.log("`cfg` dict must contain the key `name`.")
			return None
		
		cfg_    = deepcopy(cfg)
		name    = cfg_.pop("name")
		cfg_   |= kwargs
		return self.build(name=name, **cfg_)
	
	def build_from_dictlist(
		self, cfgs: Optional[list[Union[dict, Munch]]], **kwargs
	) -> Optional[list[object]]:
		"""Factory command to create detections of classes. This method gets the
		appropriate classes from the registry while passing in the parameters
		given in `cfgs`.

		Args:
			cfgs (list[dict, Munch], optional):
				List of class objects' configs.

		Returns:
			detections (list[object], optional):
				Instances of the classes that are created.
		"""
		if cfgs is None:
			return None
		
		if (not is_list_of(cfgs, expected_type=dict)
			and not is_list_of(cfgs, expected_type=Munch)):
			error_console.log("`cfgs` must be a list of dict.")
			return None
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(name=name, **cfg))
		
		return instances if len(instances) > 0 else None


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
