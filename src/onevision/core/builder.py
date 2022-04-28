#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from onevision.core.factory import Factory
from onevision.core.factory import OptimizerFactory
from onevision.core.factory import SchedulerFactory

# MARK: - Layers

ACT_LAYERS                  = Factory(name="act_layers")
ATTN_LAYERS                 = Factory(name="attn_layers")
ATTN_POOL_LAYERS            = Factory(name="attn_pool_layers")
BOTTLENECK_LAYERS           = Factory(name="bottleneck_layers")
CONV_LAYERS                 = Factory(name="conv_layers")
CONV_ACT_LAYERS             = Factory(name="conv_act_layers")
CONV_NORM_ACT_LAYERS        = Factory(name="conv_norm_act_layers")
DROP_LAYERS                 = Factory(name="drop_layers")
EMBED_LAYERS                = Factory(name="embed_layers")
HEADS 	                    = Factory(name="heads")
LINEAR_LAYERS               = Factory(name="linear_layers")
MLP_LAYERS                  = Factory(name="mlp_layers")
NORM_LAYERS                 = Factory(name="norm_layers")
NORM_ACT_LAYERS             = Factory(name="norm_act_layers")
PADDING_LAYERS              = Factory(name="padding_layers")
PLUGIN_LAYERS               = Factory(name="plugin_layers")
POOL_LAYERS                 = Factory(name="pool_layers")
RESIDUAL_BLOCKS             = Factory(name="residual_blocks")
SAMPLING_LAYERS             = Factory(name="sampling_layers")


# MARK: - Models

BACKBONES                   = Factory(name="backbones")
CALLBACKS                   = Factory(name="callbacks")
INFERENCES                  = Factory(name="inferences")
LOGGERS                     = Factory(name="loggers")
LOSSES                      = Factory(name="losses")
METRICS                     = Factory(name="metrics")
MODULE_WRAPPERS             = Factory(name="module_wrappers")
NECKS 	                    = Factory(name="necks")
OPTIMIZERS                  = OptimizerFactory(name="optimizers")
SCHEDULERS                  = SchedulerFactory(name="schedulers")

ACTION_DETECTION            = Factory(name="action_detection")
DEBLUR                      = Factory(name="deblur")
DEHAZE                      = Factory(name="dehaze")
DENOISE                     = Factory(name="denoise")
DERAIN                      = Factory(name="derain")
DESNOW                      = Factory(name="desnow")
IMAGE_CLASSIFICATION        = Factory(name="image_classification")
IMAGE_ENHANCEMENT           = Factory(name="image_enhancement")
LOW_LIGHT_IMAGE_ENHANCEMENT = Factory(name="low_light_image_enhancement")
MODELS                      = Factory(name="models")
OBJECT_DETECTION            = Factory(name="object_detection")
SUPER_RESOLUTION            = Factory(name="super_resolution")


# MARK: - Misc

AUGMENTS                    = Factory(name="augments")
DATAMODULES                 = Factory(name="datamodules")
DATASETS                    = Factory(name="datasets")
DISTANCES                   = Factory(name="distance_functions")
FILE_HANDLERS               = Factory(name="file_handler")
LABEL_HANDLERS              = Factory(name="label_handlers")
MOTIONS                     = Factory(name="motions")
TRANSFORMS                  = Factory(name="transforms")
