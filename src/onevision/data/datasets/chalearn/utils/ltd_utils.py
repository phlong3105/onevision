#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import glob
import os.path
import random
from shutil import copyfile

import cv2
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from onevision.core import progress_bar
from onevision.core import VisionBackend
from onevision.imgproc import box_xyxy_to_cxcywh_norm
from onevision.io import create_dirs
from onevision.io import load
from onevision.io import read_image
from onevision.utils import datasets_dir

__all__ = [
	"convert_single_label_file",
	"convert_yolo_labels",
	"convert_yolo_labels_asynchronous",
	"read_pkl_results",
	"shuffle_images_labels",
]

id_map    = {
	"human"     : 0,
	"bicycle"   : 1,
	"motorcycle": 2,
	"vehicle"   : 3,
}
color_map = {
	"human"     : [0, 255, 0],
	"bicycle"   : [0, 0, 255],
	"motorcycle": [0, 255, 255],
	"vehicle"   : [255, 0, 0],
}


# MARK: - Functional

def draw_rect(image, boxes) -> np.ndarray:
	image = image.copy()
	cords = boxes[:, 2:6]
	cords = cords.reshape(-1, 4)
	for i, cord in enumerate(cords):
		color    = color_map[boxes[i, 1]]
		pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
		pt1      = int(pt1[0]), int(pt1[1])
		pt2      = int(pt2[0]), int(pt2[1])
		image    = cv2.rectangle(image.copy(), pt1, pt2, color, int(max(image.shape[:2]) / 200))
	return image


def convert_single_label_file(
	image_file: str,
	save_image: bool = False,
	verbose   : bool = False,
):
	vis_file        = image_file.replace("images", "visualize")
	label_file      = image_file.replace("images", "annotations")
	label_file      = label_file.replace("image_", "annotations_")
	label_file      = label_file.replace(".jpg", ".txt")
	yolo_label_file = image_file.replace("images", "yolo")
	yolo_label_file = yolo_label_file.replace(".jpg", ".txt")
	
	# NOTE: Read images and labels
	image   = read_image(image_file, backend=VisionBackend.CV)  # RGB
	h, w, c = image.shape
	labels  = []
	with open(label_file, "r") as f:
		for l in f.read().splitlines():
			l    = l.split(" ")
			l[1] = id_map[l[1]]
			labels.append(np.array([l[0], l[1], l[2], l[3], l[4], l[5], l[6]]))
		if len(labels) == 0:
			return
		labels = np.array(labels, np.float32)
		
	# NOTE: Visualize
	if verbose:
		drawing = draw_rect(image, labels)
		drawing = drawing[:, :, ::-1]
		cv2.imshow("image", drawing)
		if save_image:
			create_dirs([os.path.dirname(vis_file)])
			cv2.imwrite(vis_file, drawing)
		cv2.waitKey(1)
	
	# NOTE: Convert to yolo format
	create_dirs([os.path.dirname(yolo_label_file)])
	with open(yolo_label_file, mode="w") as f:
		labels[:, 2:6] = box_xyxy_to_cxcywh_norm(labels[:, 2:6], h, w)
		for l in labels:
			f.write(f"{int(l[1])} {l[2]} {l[3]} {l[4]} {l[5]}\n")


def convert_yolo_labels(
	split     : str = "train",
	save_image: bool = False,
	verbose   : bool = False,
):
	image_pattern = os.path.join(
		datasets_dir, "chalearn", "ltd", split, "*", "images", "*", "*", "*.jpg"
	)
	with progress_bar() as pbar:
		for image_file in pbar.track(
			glob.glob(image_pattern),
			description=f"[bright_yellow]Converting labels"
		):
			convert_single_label_file(
				image_file=image_file, save_image=save_image, verbose=verbose
			)
				

def convert_yolo_labels_asynchronous(
	split     : str = "train",
	save_image: bool = False,
	verbose   : bool = False,
):
	image_pattern = os.path.join(
		datasets_dir, "chalearn", "ltd", split, "*", "images", "*", "*", "*.jpg"
	)
	image_files   = glob.glob(image_pattern)
	with progress_bar() as pbar:
		total = len(image_files)
		task  = pbar.add_task(f"[bright_yellow]Converting labels", total=total)
		
		def process_label_file(
			index     : int,
			save_image: bool = False,
			verbose   : bool = False,
		):
			convert_single_label_file(
				image_file = image_files[index],
				save_image = save_image,
				verbose    = verbose
			)
			pbar.update(task, advance=1)
		
		Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
			delayed(process_label_file)(i, save_image, verbose) for i in range(total)
		)


def read_pkl_results():
	pkl_file = os.path.join(datasets_dir, "chalearn", "ltd", "toolkit", "sample_predictions.pkl")
	txt_file = pkl_file.replace(".pkl", ".txt")
	contents = load(path=pkl_file)
	print(contents["Apr"])
	with open(txt_file, "w") as w:
		# w.write()
		pass


def shuffle_images_labels(split: str = "train"):
	"""Shuffle images and labels for training."""
	label_pattern = os.path.join(
		datasets_dir, "chalearn", "ltd", split, "month", "yolo", "*", "*", "*.txt"
	)
	label_files = glob.glob(label_pattern)
	random.shuffle(label_files)
	image_files = [i.replace("yolo", "images") for i in label_files]
	image_files = [l.replace(".txt", ".jpg")   for l in image_files]
	if len(image_files) != len(label_files):
		raise RuntimeError(f"Number of `image_files` and `label_files` must be "
		                   f"the same. "
		                   f"But got: {len(image_files)} != {len(label_files)}.")
	
	with progress_bar() as pbar:
		for i, (in_image, in_label)  in pbar.track(
			enumerate(zip(image_files, label_files)),
			description=f"[bright_yellow]Shuffling and copying images and labels"
		):
			i         = f"{i}".zfill(8)
			out_image = os.path.join(
				datasets_dir, "chalearn", "ltd", "extra", split, "images", f"{i}.jpg"
			)
			out_label = os.path.join(
				datasets_dir, "chalearn", "ltd", "extra", split, "labels", f"{i}.txt"
			)
			create_dirs(paths=[os.path.dirname(out_image)])
			create_dirs(paths=[os.path.dirname(out_label)])
			copyfile(in_image, out_image)
			copyfile(in_label, out_label)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--split",      default="train", type=str)
	parser.add_argument("--convert",    default=True,    action="store_true", help="Convert labels to YOLO format.")
	parser.add_argument("--save_image", default=False,   action="store_true", help="Should save image with drawn bounding boxes.")
	parser.add_argument("--verbose",    default=False,   action="store_true", help="Should show image with drawn bounding boxes.")
	parser.add_argument("--asynch",     default=True,    action="store_true", help="Run labels conversion in asynchronous mode.")
	parser.add_argument("--shuffle",    default=True,    action="store_true", help="Shuffle and copy images and labels to train/val folders.")
	args = parser.parse_args()
	
	if args.convert:
		if args.asynch:
			convert_yolo_labels_asynchronous(
				split      = args.split,
				save_image = args.save_image,
				verbose    = args.verbose,
			)
		else:
			convert_yolo_labels(
				split      = args.split,
				save_image = args.save_image,
				verbose    = args.verbose,
			)
	if args.shuffle:
		shuffle_images_labels(split=args.split)
