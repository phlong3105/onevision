<div align="center">

OneVision <img src="data/logo/one_32.png" width="32"> : One Vision Library To Rule Them All
=============================
`onevision` is a foundational library for computer vision research. 
It includes supporting functions, data types, data classes, layers, losses, 
metrics, ..., datasets, and models.

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#contact">Contact</a>
</p>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![Conda](https://img.shields.io/conda/v/conda-forge/pytorch-lightning?label=conda&color=success)](https://anaconda.org/conda-forge/pytorch-lightning)
</div>


## <div align="center">Installation</div>

<details open>
<summary>Prerequisite</summary>

- OS: [**Ubuntu 20.04 / 22.04**](https://ubuntu.com/download/desktop) (fully supports), `Windows 10` and `MacOS` (partially supports).
- Environment: 
  [**Python>=3.9.0**](https://www.python.org/),
  [**PyTorch>=1.11.0**](https://pytorch.org/get-started/locally/), 
  [**cudatoolkit=11.3**](https://pytorch.org/get-started/locally/),
  with [**anaconda**](https://www.anaconda.com/products/distribution).
- Editor: [**PyCharm**](https://www.jetbrains.com/pycharm/download).
</details>

<details open>
<summary>Directory</summary>

```text
one                   # root directory
 |__ datasets         # contains raw data
 |__ onevision        
 |__ projects
 |      |__ project1
 |      |__ project2
 |      |__ ..
 |
 |__ tools
```
</details>

<details open>
<summary>Installation using Docker (will be uploaded later)</summary>

```shell
nvidia-docker run --name one -it \ 
-v /your-datasets-path/:/datasets/ \
-v /your-projects-path/:/projects/ \
--shm-size=64g phlong/onevision
```
</details>

<details open>

<summary>Installation using conda</summary>

```shell
cd <to-where-you-want-to-save-one-dir>
mkdir -p one
mkdir -p one/datasets
cd one

# Install `aic22_track4` package
git clone git@github.com:phlong3105/onevision
cd onevision/install
chmod +x install.sh
conda init bash

# Install package. When prompt to input the dataset directory path, you should 
# enter: <some-path>/one/datasets
bash -i install.sh
cd ..
pip install --upgrade -e .
```
</details>


## <div align="center">How To Use</div>
To be updated.


## <div align="center">Citation</div>
If you find our work useful, please cite the following:

```text
@misc{Pham2022,  
    author       = {Long Hoang Pham},  
    title        = {OneVision: One Vision Library To Rule Them All},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {\url{https://github.com/phlong3105/onevision}},
    year         = {2022},
}
```


## <div align="center">Contact</div>
If you have any questions, feel free to contact `Long Pham` 
([longpham3105@gmail](longpham3105@gmail) or [phlong@skku.edu](phlong@skku.edu))
