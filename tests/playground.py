#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import ffmpeg

stream = ffmpeg.input("../data/demo.mp4")
stream = ffmpeg.hflip(stream)
stream = ffmpeg.output(stream, "../data/output.mp4")
ffmpeg.run(stream)
