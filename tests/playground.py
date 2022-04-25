#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import cv2

from onevision import FFmpegVideoLoader
from onevision import FFmpegVideoWriter

video_loader = FFmpegVideoLoader(data="../data/demo.mp4")
video_writer = FFmpegVideoWriter(dst="../data/results.mp4",	shape=video_loader.shape)

for imgs, idxes, files, rel_paths in video_loader:
	for img in imgs:
		cv2.imshow("Image", img)
		video_writer.write(img)
		if cv2.waitKey(1) == 27:
			video_loader.close()
			cv2.destroyAllWindows()
			break
