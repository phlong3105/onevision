#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os

from munch import Munch
from pytorch_lightning.plugins import DDPPlugin

from onevision import datasets_dir
from onevision import Phase
from scripts import configs

hosts = {
	"lp-desktop-windows": Munch(
		phase       = Phase.TRAINING,
		amp_backend = "apex",
		strategy    = "dp",  # DDPPlugin(find_unused_parameters=True),
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, ""),
		config      = configs.zerodce_lol199
	),
	"lp-labdesktop01-windows": Munch(
		phase       = Phase.TRAINING,
		amp_backend = "apex",
		strategy    = DDPPlugin(find_unused_parameters=True),
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, "iec", "iec22", "train", "low"),
		config      = configs.hinet_dehaze_a2i2hazeextra
	),
	"lp-labdesktop02-ubuntu": Munch(
		phase       = Phase.TRAINING,
		amp_backend = "native",
		strategy    = DDPPlugin(find_unused_parameters=True),
		gpus        = [0],
		infer_data  = os.path.join(datasets_dir, ""),
		config      = configs
	),
	"vsw-server02-ubuntu": Munch(
		phase       = Phase.TRAINING,
		amp_backend = "native",
		strategy    = DDPPlugin(find_unused_parameters=True),
		gpus        = [0, 1],
		infer_data  = os.path.join(datasets_dir, ""),
		config      = configs
	),
	"vsw-server03-ubuntu": Munch(
		phase       = Phase.TRAINING,
		amp_backend = "native",
		strategy    = DDPPlugin(find_unused_parameters=True),
		gpus        = [0, 1],
		infer_data  = os.path.join(datasets_dir, ""),
		config      = configs
	),
}
