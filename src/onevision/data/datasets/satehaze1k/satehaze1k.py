#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SateHaze1K dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt

from onevision.core import Augment_
from onevision.core import console
from onevision.core import DATAMODULES
from onevision.core import DATASETS
from onevision.core import Int3T
from onevision.core import progress_bar
from onevision.core import Tasks
from onevision.core import VisionBackend
from onevision.data.data_class import ImageInfo
from onevision.data.data_class import VisionData
from onevision.data.datamodule import DataModule
from onevision.data.dataset import ImageEnhancementDataset
from onevision.data.label_handler import VisionDataHandler
from onevision.imgproc import show_images
from onevision.core import ModelState
from onevision.utils import datasets_dir

__all__ = [
    "SateHaze1K",
    "SateHaze1KDataModule",
]


# MARK: - SateHaze1K

@DATASETS.register(name="satehaze1k")
class SateHaze1K(ImageEnhancementDataset):
    """The new haze satellite dataset on which we evaluate our approach contains
    1200 individual pairs of hazy images, corresponding hazy-free images and SAR
    images. In order to guarantee the facticity, abundance, and diversity of
    haze masks in our dataset, we use Photoshop Software to extract real haze
    masks of the easily accessible original hazy remote sensing images to
    generate transmission maps for synthetic images. The dataset consists of 3
    levels of fog, called Thin fog, Moderate fog, Thick fog. In the synthetic
    images covered by thin fog, the haze mask will be only mist which picks up
    from the original real cloudy image. For the moderate fog image, samples
    overlap with mist and medium fog. But for the thick fog, the transmission
    maps are selected from the dense haze.
    
    Training, validation and test folds. Our training, validation and test folds
    were approximately 80%, 10%, 10% of the total data respectively. We split
    every 400 images to train, valid, and test set, and artificially label 45
    of thick fog images for segmentation purposes.
    
    Args:
        level (Tasks, optional):
            The level of fog to be used. One of the values in `self.levels`.
            Can also be a list to include multiple subsets. When `all`, `*`, or
            `None`, all subsets will be included. Default: `*`.
    """
    
    levels = ["thin", "moderate", "thick"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str                     = "train",
        level           : Optional[Tasks]         = "*",
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
        self.level = level
        
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
    def level(self) -> list[str]:
        return self._level
    
    @level.setter
    def level(self, level: Optional[Tasks]):
        level = [level] if isinstance(level, str) else level
        if level is None or "all" in level or "*" in level:
            level = self.levels
        self._level = level
    
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"RESIDE ITS dataset only supports `split`: "
                             f"`train`, `val`, or `test`. Get: {self.split}.")
        
        # NOTE: List all files
        if "thin" in self.levels:
            self.list_thin_files()
        if "moderate" in self.levels:
            self.list_thin_files()
        if "thick" in self.levels:
            self.list_thin_files()
            
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
            len(self.image_paths) <= 0
            or len(self.image_paths) != len(self.eimage_paths)
        ):
            raise ValueError(
                f"Number of images != Number of enhanced images: "
                f"{len(self.image_paths)} != {len(self.eimage_paths)}."
            )
        console.log(f"Number of images: {len(self.image_paths)}.")
    
    def list_thin_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                self.root, "thin", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Thin {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
                
    def list_moderate_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                self.root, "moderate", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Moderate {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
                
    def list_thick_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                self.root, "thick", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Thick {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
 
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
        

# MARK: - SateHaze1KDataModule

@DATAMODULES.register(name="satehaze1k")
class SateHaze1KDataModule(DataModule):
    """SateHaze1K DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "satehaze1k"),
        name       : str = "satehaze1k",
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
    
    def setup(self, phase: Optional[ModelState] = None):
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
            phase (ModelState, optional):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]RESIDE ITS[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            self.train = SateHaze1K(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.val = SateHaze1K(
                root=self.dataset_dir, split="val", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",  None)
        
        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = SateHaze1K(
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
        "name": "satehaze1k",
        # Dataset's name.
        "level": "*",
        # The level of fog to be used. One of: ["thin", "moderate", "thick"].
        # Can also be a list to include multiple subsets. When `all`, `*`, or
        # `None`, all subsets will be included. Default: `*`.
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
    dm   = SateHaze1KDataModule(**cfgs)
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
