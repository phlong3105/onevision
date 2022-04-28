#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom data types.
"""

from __future__ import annotations

import functools
import types
from typing import Any
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

# MARK: - Templates
# Template for arguments which can be supplied as a tuple, or which can be a
# scalar which PyTorch will internally broadcast to a tuple. Comes in several
# variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d
# operations.
T                      = TypeVar("T")
ScalarOrTuple1T        = Union[T, tuple[T]]
ScalarOrTuple2T        = Union[T, tuple[T, T]]
ScalarOrTuple3T        = Union[T, tuple[T, T, T]]
ScalarOrTuple4T        = Union[T, tuple[T, T, T, T]]
ScalarOrTuple5T        = Union[T, tuple[T, T, T, T, T]]
ScalarOrTuple6T        = Union[T, tuple[T, T, T, T, T, T]]
ScalarOrTupleAnyT      = Union[T, tuple[T, ...]]
ScalarListOrTuple1T    = Union[T, list[T], tuple[T]]
ScalarListOrTuple2T    = Union[T, list[T], tuple[T, T]]
ScalarListOrTuple3T    = Union[T, list[T], tuple[T, T, T]]
ScalarListOrTuple4T    = Union[T, list[T], tuple[T, T, T, T]]
ScalarListOrTuple5T    = Union[T, list[T], tuple[T, T, T, T, T]]
ScalarListOrTuple6T    = Union[T, list[T], tuple[T, T, T, T, T, T]]
ScalarListOrTupleAnyT  = Union[T, list[T], tuple[T, ...]]
ScalarOrCollectionAnyT = Union[T, list[T], tuple[T, ...], dict[Any, T]]
ListOrTupleAnyT        = Union[   list[T], tuple[T, ...]]
ListOrTuple1T          = Union[   list[T], tuple[T]]
ListOrTuple2T          = Union[   list[T], tuple[T, T]]
ListOrTuple3T          = Union[   list[T], tuple[T, T, T]]
ListOrTuple4T          = Union[   list[T], tuple[T, T, T, T]]
ListOrTuple5T          = Union[   list[T], tuple[T, T, T, T, T]]
ListOrTuple6T          = Union[   list[T], tuple[T, T, T, T, T, T]]


# MARK: - Basic Types

Array1T         = ScalarListOrTuple1T[np.ndarray]
Array2T         = ScalarListOrTuple2T[np.ndarray]
Array3T         = ScalarListOrTuple3T[np.ndarray]
Array4T         = ScalarListOrTuple4T[np.ndarray]
Array5T         = ScalarListOrTuple5T[np.ndarray]
Array6T         = ScalarListOrTuple6T[np.ndarray]
ArrayAnyT       = ScalarListOrTupleAnyT[np.ndarray]
ArrayList       = list[np.ndarray]
Arrays          = ScalarOrCollectionAnyT[np.ndarray]
                
Callable        = Union[str, type, object, types.FunctionType, functools.partial]
Color           = ListOrTuple3T[int]
Devices         = Union[ScalarListOrTupleAnyT[int], ScalarListOrTupleAnyT[str]]

Int1T           = ScalarListOrTuple1T[int]
Int2T           = ScalarListOrTuple2T[int]
Int3T           = ScalarListOrTuple3T[int]
Int4T           = ScalarListOrTuple4T[int]
Int5T           = ScalarListOrTuple5T[int]
Int6T           = ScalarListOrTuple6T[int]
IntAnyT         = ScalarListOrTupleAnyT[int]
Int2Or3T        = Union[Int2T, Int3T]

Float1T         = ScalarListOrTuple1T[float]
Float2T         = ScalarListOrTuple2T[float]
Float3T         = ScalarListOrTuple3T[float]
Float4T         = ScalarListOrTuple4T[float]
Float5T         = ScalarListOrTuple5T[float]
Float6T         = ScalarListOrTuple6T[float]
FloatAnyT       = ScalarListOrTupleAnyT[float]

Indexes         = ScalarListOrTupleAnyT[int]
Number          = Union[int, float]

Padding1T       = Union[ScalarListOrTuple1T[int],   str]
Padding2T       = Union[ScalarListOrTuple2T[int],   str]
Padding3T       = Union[ScalarListOrTuple3T[int],   str]
Padding4T       = Union[ScalarListOrTuple4T[int],   str]
Padding5T       = Union[ScalarListOrTuple5T[int],   str]
Padding6T       = Union[ScalarListOrTuple6T[int],   str]
PaddingAnyT     = Union[ScalarListOrTupleAnyT[int], str]

Tensor1T        = ScalarListOrTuple1T[Tensor]
Tensor2T        = ScalarListOrTuple2T[Tensor]
Tensor3T        = ScalarListOrTuple3T[Tensor]
Tensor4T        = ScalarListOrTuple4T[Tensor]
Tensor5T        = ScalarListOrTuple5T[Tensor]
Tensor6T        = ScalarListOrTuple6T[Tensor]
TensorAnyT      = ScalarListOrTupleAnyT[Tensor]
TensorList      = list[Tensor]
TensorOrArray   = Union[Tensor, np.ndarray]
Tensors         = ScalarOrCollectionAnyT[Tensor]
TensorsOrArrays = Union[Tensors, Arrays]


# MARK: - Model's Parameters

Config        = Union[str, dict, list]
Losses_       = Union[_Loss,     list[Union[_Loss,     dict]], dict]
Metrics_      = Union[Metric,    list[Union[Metric,    dict]], dict]
Optimizers_   = Union[Optimizer, list[Union[Optimizer, dict]], dict]

LabelTypes    = ScalarListOrTupleAnyT[str]
Metrics       = Union[dict[str, Tensor], dict[str, np.ndarray]]
Pretrained    = Union[bool, str, dict]
Tasks         = ScalarListOrTupleAnyT[str]

ForwardOutput = tuple[Tensors, Optional[Tensor]]
StepOutput    = Union[Tensor, dict[str, Any]]
EpochOutput   = list[StepOutput]
EvalOutput    = list[dict[str, float]]
PredictOutput = Union[list[Any], list[list[Any]]]
Weights       = Union[Tensor, ListOrTupleAnyT[float], ListOrTupleAnyT[int]]


# MARK: - Data / Dataset / Datamodule

Augment_         = Union[dict, Callable]
Transform_       = Union[dict, Callable]
Transforms_      = Union[str, nn.Sequential, Transform_, list[Transform_]]
TrainDataLoaders = Union[
    DataLoader,
    Sequence[DataLoader],
    Sequence[Sequence[DataLoader]],
    Sequence[dict[str, DataLoader]],
    dict[str, DataLoader],
    dict[str, dict[str, DataLoader]],
    dict[str, Sequence[DataLoader]],
]
EvalDataLoaders = Union[DataLoader, Sequence[DataLoader]]
