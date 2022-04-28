#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
                              _
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||_  \
                   |   | \\\  -  /'| |   |
                   | \_|  `\`---'//  |_/ |
                   \  .-\__ `-. -'__/-.  /
                 ___`. .'  /--.--\  `. .'___
              ."" '<  `.___\_<|>_/___.' _> \"".
             | | :  `- \`. ;`. _/; .'/ /  .' ; |
             \  \ `-.   \_\_`. _.'_/_/  -' _.' /
   ===========`-.`___`-.__\ \___  /__.-'_.'_.-'================
                           `=--=-'
"""

from __future__ import annotations

import os
from shutil import copyfile

from munch import Munch

from .core import *
from .data import *
from .imgproc import *
from .io import *
from .math import *
from .models import *
from .nn import *
from .utils import *

__author__  = "Long H. Pham"
__version__ = "0.2.0"


# MARK: - Process Config

def load_config(config: Union[str, dict]) -> Munch:
	"""Load config as namespace.

	Args:
		config (str, dict):
			Config filepath that contains configuration values or the
			config dict.
	"""
	# NOTE: Load dictionary from file and convert to namespace using Munch
	if isinstance(config, str):
		config_dict = load(path=config)
	elif isinstance(config, dict):
		config_dict = config
	else:
		raise ValueError
	
	assert (config_dict is not None), f"No configuration is found at {config}!"
	config = Munch.fromDict(config_dict)
	return config


def copy_config_file(config_file: str, dst: str):
	"""Copy config file to destination dir."""
	create_dirs(paths=[dst])
	copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))
