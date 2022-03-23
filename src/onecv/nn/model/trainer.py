#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Add-in to the `pytorch_lightning.Trainer` class.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.utilities import _IPU_AVAILABLE
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities import DeviceType

from onecv.utils import console
from onecv.utils import error_console

__all__ = [
    "Trainer"
]


# MARK: - Trainer

class Trainer(pl.Trainer):
    """Override `pytorch_lightning.Trainer` with several methods and properties.
    """
    
    # MARK: Properties
    
    @pl.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        self.fit_loop.current_epoch = current_epoch
    
    @pl.Trainer.global_step.setter
    def global_step(self, global_step: int):
        self.fit_loop.global_step = global_step
        
    # MARK: Configure

    def _log_device_info(self):
        console.log(f"GPU available: {torch.cuda.is_available()}, "
                    f"used: {self._device_type == DeviceType.GPU}")

        num_tpu_cores = self.tpu_cores if (
	        self.tpu_cores is not None and self._device_type == DeviceType.TPU
        ) else 0
        console.log(f"TPU available: {_TPU_AVAILABLE}, "
                    f"using: {num_tpu_cores} TPU cores")

        num_ipus = self.ipus if self.ipus is not None else 0
        console.log(f"IPU available: {_IPU_AVAILABLE}, "
                    f"using: {num_ipus} IPUs")

        if torch.cuda.is_available() and self._device_type != DeviceType.GPU:
            error_console.log(
                "GPU available but not used. Set the gpus flag in your trainer "
                "`Trainer(gpus=1)` or script `--gpus=1`."
            )

        if _TPU_AVAILABLE and self._device_type != DeviceType.TPU:
            error_console.log(
                "TPU available but not used. Set the `tpu_cores` flag in your "
                "trainer `Trainer(tpu_cores=8)` or script `--tpu_cores=8`."
            )

        if (
	        _IPU_AVAILABLE
	        and self._device_type != DeviceType.IPU
	        and not isinstance(self.accelerator, IPUAccelerator)
        ):
            error_console.log(
                "IPU available but not used. Set the `ipus` flag in your "
                "trainer `Trainer(ipus=8)` or script `--ipus=8`."
            )
