#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
                              _
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---"\____
                     ."  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||_  \
                   |   | \\\  -  /"| |   |
                   | \_|  `\`---"//  |_/ |
                   \  .-\__ `-. -"__/-.  /
                 ___`. ."  /--.--\  `. ."___
              ."" "<  `.___\_<|>_/___." _> \"".
             | | :  `- \`. ;`. _/; ."/ /  ." ; |
             \  \ `-.   \_\_`. _."_/_/  -" _." /
   ===========`-.`___`-.__\ \___  /__.-"_."_.-"================
                           `=--=-"

How to install:
    pip install -e ./ --upgrade
"""

from __future__ import annotations

import glob
import os.path
import pathlib

from setuptools import find_packages
from setuptools import setup

current_dir = os.path.dirname(os.path.abspath(__file__))
here        = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="onecv",
    version="0.1.0",
    description="'One Computer Vision Framework' to rule them all",
    license="LICENSE.txt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phlong3105/onecv",
    author="Long H. Pham",
    author_email="longpham3105@gmail.com",
    
    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",

        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by "pip install". See instead "python_requires" below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],

    keywords="core, data type, factory, builder, neural network, layers, computer vision",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9, <4",

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=[
        "cython",
        "filterpy",
        "koila",
        "labelme",
        "Markdown",
        "matplotlib",
        "mpi4py; sys_platform=='win32' or sys_platform=='linux'",
        "md-mermaid",
        "munch",
        "mypy",
        "multipledispatch",
        "ninja",
        "numpy",
        "opencv-python==4.4.0.46                 ;sys_platform=='win32' or sys_platform=='linux'",
        "opencv-contrib-python==4.4.0.46         ;sys_platform=='win32' or sys_platform=='linux'",
        "opencv-python-headless==4.4.0.46        ;sys_platform=='darwin'",
        "opencv-contrib-python-headless==4.4.0.46;sys_platform=='darwin'",
        "ordered-enum",
        "pandas",
        "Pillow",
        "piq",
        "protobuf",
        "pycocotools>=2.0.4",
        "pydot",
        "PyQt5",
        "pytorch-lightning>=1.5.10",
        "pyvips;sys_platform=='darwin' or sys_platform=='linux'",
        "PyYAML",
        "pretrainedmodels",
        "ptflops",
        "rawpy",
        "requests",
        "rich",
        "scikit-learn",
        "scipy",
        "seaborn",
        "sortedcontainers",
        "setGPU",
        "tensorboard",
        "thop",
        "timm",
        "torch==1.11.0+cu113      ; sys_platform=='win32' or sys_platform=='linux'",
        "torchvision==0.12.0+cu113; sys_platform=='win32' or sys_platform=='linux'",
        "torchaudio==0.11.0+cu113 ; sys_platform=='win32' or sys_platform=='linux'",
        "torch==1.11.0            ; sys_platform=='darwin'",
        "torchvision==0.12.0      ; sys_platform=='darwin'",
        "torchaudio==0.11.0       ; sys_platform=='darwin'",
        "torchmetrics[all]",
        "torchsummary",
        "tqdm",
        "validators",
        "xmltodict"
    ],
    
    data_files=[
        ("data", glob.glob(os.path.join(current_dir, "data", "*"), recursive=True))
    ],
    
    # List additional URLs that are relevant to your project as a dict.
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={
        "Bug Reports": "https://github.com/phlong3105/onecv/issues",
        "Source": "https://github.com/phlong3105/onecv/",
    },
)
