#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import sys

root = os.path.dirname(os.path.dirname(__file__))  # workspaces/one/onecv
onecv_src_root = os.path.join(root, "src")
if str(onecv_src_root) not in sys.path:
    sys.path.append(str(onecv_src_root))  # add ROOT to PATH

print(os.environ["DATASETS_DIR"])
