#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Clear all .cache
"""

from __future__ import annotations

import os.path

from onevision import datasets_dir
from onevision import delete_files


# MARK: - Functional

delete_files(
	dirpaths=[os.path.join(datasets_dir, "*")],
	extension=".cache",
	recursive=True
)
