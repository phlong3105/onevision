#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os

from pathlib import Path

full_path = "workspaces/one/projects/aic/src/aic"
print(str(Path(full_path).parents[0]))  # "path/to"
print(str(Path(full_path).parents[1]))  # "path"
print(str(Path(full_path).parents[3]))  # "."
