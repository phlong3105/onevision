#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HIDE Blur dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
from torch.utils.data import random_split

from onevision.core import Augment_
from onevision.core import DATAMODULES
from onevision.core import DATASETS
from onevision.core import Int3T
from onevision.core import Tasks
from onevision.core import VisionBackend
from onevision.data.data_class import ImageInfo
from onevision.data.data_class import VisionData
from onevision.data.datamodule import DataModule
from onevision.data.dataset import ImageEnhancementDataset
from onevision.data.label_handler import VisionDataHandler
from onevision.imgproc import show_images
from onevision.nn import Phase
from onevision.utils import console
from onevision.utils import datasets_dir
from onevision.utils import progress_bar

__all__ = [
    "HIDEBlur",
    "HIDEBlurDataModule",
]


# MARK: - HIDEBlur

@DATASETS.register(name="hideblur")
class HIDEBlur(ImageEnhancementDataset):
    """A new blurry image dataset (HIDE) with respect to the dynamic deblurring
    problem. The multiple blurs caused by the relative movement between an
    imaging device and a scene, mainly include camera shaking and object
    movement. To fully capture the dynamic blurs caused by the passive device
    interference and initiative actions, our HIDE dataset is eleborately
    collected to cover both wide-range and close-range scenes and address
    human-aware motion deblurring.
    
    We release the HIDE dataset with the blurry and sharp image pair, it could
    be downloaded from HIDE_dataset. The annotations of the HIDE dataset in
    terms of the depths (long-shot/close-up) and quantity of
    human (scattered/crowded) as well be provided. Please feel free to download.

    Args:
        depth (Tasks, optional):
            Depths (long-shot/close-up). One of the values in `self.depths`.
            Can also be a list to include multiple subsets. When `all`, `*`, or
            `None`, all subsets will be included. Default: `*`.
        quantity (Tasks, optional):
            Quantity of human (scattered/crowded). One of the values in
            `self.quantities`. Can also be a list to include multiple subsets.
            When `all`, `*`, or `None`, all subsets will be included.
            Default: `*`.
    """
    
    depths     = ["long_shot", "close_up"]
    quantities = ["scattered", "crowded"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        depth           : Optional[Tasks]         = "*",
        quantity        : Optional[Tasks]         = "*",
        split           : str                     = "train",
        shape           : Int3T                   = (720, 1280, 3),
        caching_labels  : bool                    = False,
        caching_images  : bool                    = False,
        write_labels    : bool                    = False,
        fast_dev_run    : bool                    = False,
        load_augment    : Optional[dict]          = None,
        augment         : Optional[Augment_]      = None,
        vision_backend  : Optional[VisionBackend] = None,
        transforms      : Optional[Callable]      = None,
        transform       : Optional[Callable]      = None,
        target_transform: Optional[Callable]      = None,
        *args, **kwargs
    ):
        self.depth    = depth
        self.quantity = quantity
        
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            caching_labels   = caching_labels,
            caching_images   = caching_images,
            write_labels     = write_labels,
            fast_dev_run     = fast_dev_run,
            load_augment     = load_augment,
            augment          = augment,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform,
            vision_backend   = vision_backend,
            *args, **kwargs
        )
    
    # MARK: Properties
    
    @property
    def depth(self) -> list[str]:
        return self._depth
    
    @depth.setter
    def depth(self, depth: Optional[Tasks]):
        depth = [depth] if isinstance(depth, str) else depth
        if depth is None or "all" in depth or "*" in depth:
            depth = self.depths
        self._depth = depth

    @property
    def quantity(self) -> list[str]:
        return self._depth

    @quantity.setter
    def quantity(self, quantity: Optional[Tasks]):
        quantity = [quantity] if isinstance(quantity, str) else quantity
        if quantity is None or "all" in quantity or "*" in quantity:
            quantity = self.quantities
        self._quantity = quantity
        
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        # NOTE: List all files
        if self.split == "train":
            self.list_train_files()
        elif self.split == "test":
            self.list_test_files()
        else:
            raise ValueError(f"HIDE Blur dataset only supports `split`: "
                             f"`train` or `test`. Get: {self.split}.")
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths        = [self.image_paths[i]        for i in indices]
            self.eimage_paths       = [self.eimage_paths[i]       for i in indices]
            # self.label_paths        = [self.label_paths[i]        for i in indices]
            self.custom_label_paths = [self.custom_label_paths[i] for i in indices]
        
        # NOTE: Assertion
        if (
            len(self.image_paths) <= 0 or
            len(self.image_paths) != len(self.eimage_paths)
        ):
            raise ValueError(
                f"Number of images != Number of enhanced images: "
                f"{len(self.image_paths)} != {len(self.eimage_paths)}."
            )
        console.log(f"Number of images: {len(self.image_paths)}.")
        
    def list_train_files(self):
        """List training files."""
        """
        filenames = []
        if "long_shot" in self.depth:
            with open(os.path.join(self.root, "depth_long_shot.txt")) as f:
                filenames += f.read().splitlines()
        if "close_up" in self.depth:
            with open(os.path.join(self.root, "depth_close_up.txt")) as f:
                filenames += f.read().splitlines()
        if "scattered" in self.quantity:
            with open(os.path.join(self.root, "quantity_scattered.txt")) as f:
                filenames += f.read().splitlines()
        if "crowded" in self.quantity:
            with open(os.path.join(self.root, "quantity_crowded.txt")) as f:
                filenames += f.read().splitlines()
        filenames    = list(set(filenames))
        training_dir = os.path.join(self.root, "train")
        """
        with progress_bar() as pbar:
            image_paths = glob.glob(os.path.join(self.root, "train", "*.png"))
        
            for image_path in pbar.track(
                image_paths,
                description=f"[bright_yellow]Listing {self.split} images"
            ):
                eimage_path       = image_path.replace("train", "gt")
                custom_label_path = image_path.replace("train", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)

    def list_test_files(self):
        """List testing files."""
        with progress_bar() as pbar:
            image_paths = []
            if "long_shot" in self.depth:
                image_paths += glob.glob(
                    os.path.join(self.root, "test", "test_long_shot", "*.png")
                )
            if "close_up" in self.depth:
                image_paths += glob.glob(
                    os.path.join(self.root, "test", "test_close_ups", "*.png")
                )
        
            image_paths = list(set(image_paths))
            gt_dir      = os.path.join(self.root, "gt")
            
            for image_path in pbar.track(
                image_paths,
                description=f"[bright_yellow]Listing {self.split} images"
            ):
                filename    = os.path.basename(image_path)
                eimage_path = os.path.join(gt_dir, filename)
                label_path  = os.path.join(self.root, "annotations_custom", filename)
                label_path  = label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                self.label_paths.append(label_path)
                self.custom_label_paths.append(label_path)
    
    # MARK: Load Data
    
    def load_label(
        self,
        image_path       : str,
        enhance_path     : str,
        label_path       : Optional[str] = None,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            enhance_path (str):
                Enhanced image file.
            label_path (str, optional):
                Label file. Default: `None`.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.
    
        Returns:
            data (VisionData):
                `VisionData` object.
        """
        # NOTE: If we have custom labels
        if custom_label_path and os.path.isfile(custom_label_path):
            return VisionDataHandler().load_from_file(
                image_path  = image_path,
                label_path  = custom_label_path,
                eimage_path = enhance_path
            )

        # NOTE: Parse info
        image_info  = ImageInfo.from_file(image_path=image_path)
        eimage_info = ImageInfo.from_file(image_path=enhance_path)
        
        return VisionData(image_info=image_info, eimage_info=eimage_info)

    def load_class_labels(self):
        """Load ClassLabels."""
        pass


# MARK: - HIDEBlurDataModule

@DATAMODULES.register(name="hideblur")
class HIDEBlurDataModule(DataModule):
    """HIDE DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "hide"),
        name       : str = "hideblur",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_kwargs = kwargs
        
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_labels is None:
            self.load_class_labels()
    
    def setup(self, phase: Optional[Phase] = None):
        """There are also data operations you might want to perform on every
        GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (Phase, optional):
                Phase to use: [None, Phase.TRAINING, Phase.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]HIDE[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, Phase.TRAINING]:
            full_dataset = HIDEBlur(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size   = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",   None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test        = HIDEBlur(
                root=self.dataset_dir, split="test", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        pass


# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "hideblur",
        # Dataset's name.
        "subset": ["blur"],
        # Sub-dataset to use. One of the values in `self.subsets`. Can also be a
        # list to include multiple subsets. When `all`, `*`, or `None`, all subsets
        # will be included. Default: `blur`.
        "shape": [512, 512, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        "batch_size": 4,
        # Number of samples in one forward & backward pass.
        "caching_labels": True,
        # Should overwrite the existing cached labels? Default: `False`.
        "caching_images": False,
        # Cache images into memory for faster training. Default: `False`.
        "write_labels": False,
        # After loading images and labels for the first time, we will convert it
        # to our custom data format and write to files. If `True`, we will
        # overwrite these files. Default: `False`.
        "fast_dev_run": False,
        # Take a small subset of the data for fast debug (i.e, like unit testing).
        # Default: `False`.
        "shuffle": True,
        # Set to `True` to have the data reshuffled at every training epoch.
        # Default: `True`.
        "load_augment": {
            "mosaic": 0.0,
            "mixup" : 0.5,
        },
        # Augmented loading policy.
        "augment": {
            "name": "paired_images_auto_augment",
            # Name of the augmentation policy.
            "policy": "enhancement",
            # Augmentation policy. One of: [`enhancement`]. Default: `enhancement`.
            "fill": None,
            # Pixel fill value for the area outside the transformed image.
            # If given a number, the value is used for all bands respectively.
            "to_tensor": True,
            # Convert a PIL Image or numpy.ndarray [H, W, C] in the range [0, 255]
            # to a torch.FloatTensor of shape [C, H, W] in the  range [0.0, 1.0].
            # Default: `True`.
        },
        # Augmentation policy.
        "vision_backend": VisionBackend.PIL,
        # Vision backend option.
    }
    dm   = HIDEBlurDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize one sample
    data_iter            = iter(dm.train_dataloader)
    input, target, shape = next(data_iter)
    show_images(images=input,  nrow=2, denormalize=True)
    show_images(images=target, nrow=2, denormalize=True, figure_num=1)
    plt.show(block=True)
