#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfile
from typing import Union

from munch import Munch

from onevision.io import create_dirs
from onevision.io import load

__all__ = [
    "content_root_dir",
    "copy_config_file",
    "datasets_dir",
    "load_config",
    "pretrained_dir",
    "source_root_dir",
]


# MARK: - Directories

__current_file   = os.path.abspath(__file__)                          # "workspaces/one/onevision/src/onevision/utils.py"
source_root_dir  = os.path.dirname(__current_file)                    # "workspaces/one/onevision/src/onevision"
content_root_dir = os.path.dirname(os.path.dirname(source_root_dir))  # "workspaces/one/onevision"
pretrained_dir   = os.path.join(source_root_dir, "pretrained")        # "workspaces/one/onevision/src/onevision/pretrained"
datasets_dir     = os.getenv("DATASETS_DIR", None)  # In case we have set value in os.environ
if datasets_dir is None:  # Run in debug mode from PyCharm
    datasets_dir = os.path.join(str(Path(source_root_dir).parents[2]), "datasets")  # "workspaces/one/datasets
if not os.path.isdir(datasets_dir):
    datasets_dir = ""
    # raise RuntimeWarning("`datasets_dir` has not been set.")


# MARK: - Process Config

def load_config(config: Union[str, dict, Munch]) -> Munch:
    """Load config as namespace.

	Args:
		config (str, dict, Munch):
			Config filepath that contains configuration values or the
			config dict.
	"""
    # NOTE: Load dictionary from file and convert to namespace using Munch
    if isinstance(config, str):
        config_dict = load(path=config)
    elif isinstance(config, (dict, Munch)):
        config_dict = config
    else:
        raise ValueError(f"`config` must be a `dict` or a path to config file. "
                         f"But got: {config}.")
    if config_dict is None:
        raise ValueError(f"No configuration is found at: {config}.")
   
    config = Munch.fromDict(config_dict)
    return config


def copy_config_file(config_file: str, dst: str):
    """Copy `config_file` to `dst` dir."""
    create_dirs(paths=[dst])
    copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))
