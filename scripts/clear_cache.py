#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Clear all .cache
"""

from __future__ import annotations

import os.path

from onevision import datasets_dir
from onevision import delete_files

if __name__ == "__main__":
	delete_files(
		dirs= [os.path.join(datasets_dir, "*")],
		extension = ".cache",
		recursive = True
	)
