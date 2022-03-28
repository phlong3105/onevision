#!/usr/bin/env python img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from onevision.cv.core import to_channel_last
from onevision.cv.io.image import is_image_file
from onevision.cv.io.video import is_video_file
from onevision.cv.io.video import is_video_stream
from onevision.file import create_dirs
from onevision.type import Arrays
from onevision.type import Int2Or3T
from onevision.type import to_4d_array
from onevision.type import to_size

__all__ = [
	"FrameLoader",
	"FrameWriter",
]


# MARK: - Modules

class FrameLoader:
	"""Frame Loader retrieves and loads frame(s) from a filepath, a pathname
	pattern, a directory, a video, or a stream.

	Attributes:
		data (str):
			Data source. Can be a path to an image file, a directory, a video,
			or a stream. It can also be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass.
		image_files (list):
			List of image files found in the data source.
		video_capture (VideoCapture):
			VideoCapture object from OpenCV.
		num_frames (int):
			Total number of image files or total number of frames in the video.
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data          = data
		self.batch_size    = batch_size
		self.image_files   = []
		self.video_capture = None
		self.num_frames    = -1
		self.index         = 0

		self.init_image_files_or_video_capture(data=self.data)

	def __len__(self):
		"""Get the number of frames in the video or the number of images in
		`image_files`.

		Returns:
			num_frames (int):
				>0 if the offline video.
				-1 if the online video.
		"""
		return self.num_frames  # number of frame, [>0 : video, -1 : online_stream]
	
	def batch_len(self) -> int:
		"""Return the total batches calculated from `batch_size`."""
		return int(self.__len__() / self.batch_size)
	
	def __iter__(self):
		"""Return an iterator starting at index 0.

		Returns:
			self (VideoInputStream):
				For definition __next__ below.
		"""
		self.index = 0
		return self

	def __next__(self):
		"""Next items.
			e.g.:
				>>> video_stream = FrameLoader("cam_1.mp4")
				>>> for image, index in enumerate(video_stream):

		Returns:
			images (np.ndarray):
				List of image file from opencv with `np.ndarray` type.
			indexes (list):
				List of image indexes.
			files (list):
				List of image files.
			rel_paths (list):
				List of images' relative paths corresponding to data.
		"""
		if self.index >= self.num_frames:
			raise StopIteration
		else:
			images    = []
			indexes   = []
			files     = []
			rel_paths = []

			for i in range(self.batch_size):
				if self.index >= self.num_frames:
					break

				if self.video_capture:
					ret_val, image = self.video_capture.read()
					rel_path 	   = os.path.basename(self.data)
				else:
					image	 = cv2.imread(self.image_files[self.index])
					file     = self.image_files[self.index]
					rel_path = file.replace(self.data, "")
				
				if image is not None:
					image = image[:, :, ::-1]  # BGR to RGB
					
				images.append(image)
				indexes.append(self.index)
				files.append(self.data)
				rel_paths.append(rel_path)

				self.index += 1

			return np.array(images), indexes, files, rel_paths

	def __del__(self):
		"""Close `video_capture` object."""
		self.close()
	
	# MARK: Configure

	def init_image_files_or_video_capture(self, data: str):
		"""Initialize image files or `video_capture` object.

		Args:
			data (str):
				Data source. Can be a path to an image file, a directory,
				a video, or a stream. It can also be a pathname pattern to
				images.
		"""
		if is_video_file(data):
			self.video_capture = cv2.VideoCapture(data)
			self.num_frames    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif is_video_stream(data):
			self.video_capture = cv2.VideoCapture(data)  # stream
			self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)  # set buffer (batch) size
			self.num_frames    = -1
		elif is_image_file(data):
			self.image_files = [data]
			self.num_frames  = len(self.image_files)
		elif os.path.isdir(data):
			self.image_files = [img for img in glob.glob(os.path.join(data, "**/*"), recursive=True) if is_image_file(img)]
			self.num_frames  = len(self.image_files)
		elif isinstance(data, str):
			self.image_files = [img for img in glob.glob(data) if is_image_file(img)]
			self.num_frames  = len(self.image_files)
		else:
			raise IOError(f"Error when reading data!")

	def close(self):
		"""Release the `video_capture` object."""
		if self.video_capture:
			self.video_capture.release()


class FrameWriter:
	"""Frame Writer saves frames to individual image files or appends all to a
	video file.

	Attributes:
		dst (str):
			Output video file or a directory.
		video_writer (VideoWriter):
			`VideoWriter` object from OpenCV.
		shape (Int3T):
			Output size as [H, W, C]. This is also used to reshape the input.
		frame_rate (int):
			Frame rate of the video.
		fourcc (str):
			Video codec. One of: ["mp4v", "xvid", "mjpg", "wmv1"].
		save_image (bool):
			Should write individual image?
		save_video (bool):
			Should write video?
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		dst		  : str,
		shape     : Int2Or3T = (480, 640, 3),
		frame_rate: float     = 10,
		fourcc    : str       = "mp4v",
		save_image: bool      = False,
		save_video: bool      = True,
	):
		super().__init__()
		self.dst		  = dst
		self.shape        = shape
		self.image_size   = to_size(shape)
		self.frame_rate   = frame_rate
		self.fourcc       = fourcc
		self.save_image   = save_image
		self.save_video   = save_video
		self.video_writer = None
		self.index		  = 0

		if self.save_video:
			self.init_video_writer()

	def __len__(self):
		"""Return the number of already written frames."""
		return self.index

	def __del__(self):
		"""Close the `video_writer`."""
		self.close()

	# MARK: Configure

	def init_video_writer(self):
		"""Initialize `video_writer` object."""
		if os.path.isdir(self.dst):
			parent_dir = self.dst
			video_file = os.path.join(parent_dir, f"result.mp4")
		else:
			parent_dir = str(Path(self.dst).parent)
			stem       = str(Path(self.dst).stem)
			video_file = os.path.join(parent_dir, f"{stem}.mp4")
		create_dirs(paths=[parent_dir])

		fourcc			  = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_writer = cv2.VideoWriter(
			video_file, fourcc, self.frame_rate, self.image_size[::-1]  # Must be in [W, H]
		)

		if self.video_writer is None:
			raise FileNotFoundError(f"Cannot create video file at: {video_file}.")

	def close(self):
		"""Close the `video_writer`."""
		if self.video_writer:
			self.video_writer.release()

	# MARK: Write

	def write_frame(self, image: np.ndarray, image_file: Optional[str] = None):
		"""Add a frame to writing video.

		Args:
			image (np.ndarray):
				Image.
			image_file (str, optional):
				Image file. Default: `None`.
		"""
		image = to_channel_last(image)

		if self.save_image:
			if image_file is not None:
				image_file = (image_file[1:] if image_file.startswith("\\")
							  else image_file)
				image_name = os.path.splitext(image_file)[0]
			else:
				image_name = f"{self.index}"
			output_file = os.path.join(self.dst, f"{image_name}.png")
			parent_dir  = str(Path(output_file).parent)
			create_dirs(paths=[parent_dir])
			cv2.imwrite(output_file, image)
		if self.save_video:
			self.video_writer.write(image)

		self.index += 1

	def write_frames(self, images: Arrays, image_files: Optional[list[str]] = None):
		"""Add batch of frames to video.

		Args:
			images (Arrays):
				Images.
			image_files (list[str], optional):
				Image files. Default: `None`.
		"""
		images = to_4d_array(images)
		
		if image_files is None:
			image_files = [None for _ in range(len(images))]

		for image, image_file in zip(images, image_files):
			self.write_frame(image=image, image_file=image_file)
