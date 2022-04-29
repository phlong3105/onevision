#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations on collections.
"""

from __future__ import annotations

import collections
import itertools
from collections import abc
from collections import OrderedDict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

from multipledispatch import dispatch

from onevision.core.globals import Int2Or3T
from onevision.core.globals import Int2T

__all__ = [
    "concat_list",
    "copy_attr",
    "intersect_dicts",
    "intersect_ordered_dicts",
    "is_dict_of",
    "is_list_of",
    "is_seq_of",
    "is_tuple_of",
    "slice_list",
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_iter",
    "to_list",
    "to_ntuple",
    "to_size",
    "to_tuple",
    "unique",
]


# MARK: - Functional

def concat_list(in_list: list) -> list:
    """Concatenate a list of list into a single list."""
    return list(itertools.chain(*in_list))


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, options to only include [...] and to
    exclude [...]."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da: dict, db: dict, exclude: Union[tuple, list] = ()) -> dict:
    """Dictionary intersection of matching keys and shapes, omitting `exclude`
    keys, using da values.
    """
    return {
        k: v for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def intersect_ordered_dicts(
    da: OrderedDict, db: OrderedDict, exclude: Union[tuple, list] = ()
) -> OrderedDict:
    """Dictionary intersection of matching keys and shapes, omitting `exclude`
    keys, using da values.
    """
    return OrderedDict(
        (k, v) for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and v.shape == db[k].shape)
    )


def is_seq_of(
    seq: Sequence, expected_type: type, seq_type: Optional[type] = None
) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence):
            Sequence to be checked.
        expected_type (type):
            Expected type of sequence items.
        seq_type (type, optional):
            Expected sequence type.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        if not isinstance(seq_type, type):
            raise TypeError(f"`seq_type` must be a valid type. But got: {seq_type}.")
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_dict_of(d: dict, expected_type: type) -> bool:
    """Check whether it is a dict of some type."""
    if not isinstance(expected_type, type):
        raise TypeError(f"`expected_type` must be a valid type. But got: {expected_type}.")
    return all(isinstance(v, expected_type) for k, v in d.items())


def is_list_of(seq: list, expected_type: type) -> bool:
    """Check whether it is a list of some type. A partial method of
    `is_seq_of()`.
    """
    return is_seq_of(seq=seq, expected_type=expected_type, seq_type=list)


def is_tuple_of(seq: tuple, expected_type: type) -> bool:
    """Check whether it is a tuple of some type. A partial method of
    `is_seq_of()`."""
    return is_seq_of(seq=seq, expected_type=expected_type, seq_type=tuple)


def slice_list(in_list: list, lens: Union[int, list]) -> list[list]:
    """Slice a list into several sub lists by a list of given length.
    
    Args:
        in_list (list):
            List to be sliced.
        lens (int, list):
            Expected length of each out list.
    
    Returns:
        out_list (list):
            A list of sliced list.
    """
    if isinstance(lens, int):
        if len(in_list) % lens != 0:
            raise ValueError(f"Length of `in_list` must be divisible by `lens`."
                             f" But got: {len(in_list)} % {lens} != 0.")
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError(f"`indices` must be an `int` or `list[int]`."
                        f" But got: {type(lens)}")
    elif sum(lens) != len(in_list):
        raise ValueError(f"Sum of `lens` and length of `in_list` must be the same."
                         f"But got: {sum(lens)} != {len(in_list)}.")
    
    out_list = []
    idx      = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def to_iter(inputs: Iterable, dst_type: type, return_type: Optional[type] = None):
    """Cast elements of an iterable object into some type.
    
    Args:
        inputs (Iterable):
            Input object.
        dst_type (type):
            Destination type.
        return_type (type, optional):
            If specified, the output object will be converted to this type,
            otherwise an iterator.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError(f"`inputs` must be an iterable object. But got: {type(inputs)}")
    if not isinstance(dst_type, type):
        raise TypeError(f"`dst_type` must be a valid type. But got {type(dst_type)}")

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def to_list(inputs: Iterable, dst_type: type):
    """Cast elements of an iterable object into a list of some type. A partial
    method of `to_iter()`.
    """
    return to_iter(inputs=inputs, dst_type=dst_type, return_type=list)


def to_ntuple(n: int):
    """A helper functions to cast input to n-tuple."""
    
    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    
    return parse


def to_size(size: Int2Or3T) -> Int2T:
    if isinstance(size, (list, tuple)):
        if len(size) == 3:
            size = size[0:2]
        if len(size) == 1:
            size = (size[0], size[0])
    elif isinstance(size, (int, float)):
        size = (size, size)
    return tuple(size)


def to_tuple(inputs: Iterable, dst_type: type):
    """Cast elements of an iterable object into a tuple of some type. A partial
    method of `to_iter()`."""
    return to_iter(inputs=inputs, dst_type=dst_type, return_type=tuple)


@dispatch(list)
def unique(in_list: list) -> list:
    """Return a list with only unique elements."""
    return list(set(in_list))


@dispatch(tuple)
def unique(in_tuple: tuple) -> tuple:
    """Return a tuple with only unique elements."""
    return tuple(set(in_tuple))


@dispatch(tuple)
def unique(in_tuple: tuple) -> tuple:
    """Return a tuple with only unique elements."""
    return tuple(set(in_tuple))


# MARK: - Alias

to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_5tuple = to_ntuple(5)
to_6tuple = to_ntuple(6)
