
<div align="center">
<img src="docs/one_32.png" width="32"> 

OneVision: One Computer Vision Library To Rule Them All
=============================
`onevision` is a foundational library for computer vision research. 
It includes supporting functions, data types, data classes, layers, losses, 
metrics, ..., datasets, and models.

<p align="center">
  <a href="https://www.pytorchlightning.ai/">Website</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#contact">contact</a> •
</p>

</div>


## <div align="center"> Installation </div>
### Prerequisite
- OS: `Ubuntu 20.04/22.04` (fully supports), `Windows 10 and MacOS` (partially supports).
- Base environment: `Python 3.9+` and `PyTorch (>= v1.11.0)` with `conda`.
- Editor: `PyCharm`.

### Directory
- Here is the directories' hierarchy:
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

### Easy Installation 
```shell
cd <to-workspace-dir>
mkdir -p one
mkdir -p one/datasets
cd one

# Install `aic22_track4` package
git clone git@github.com:phlong3105/onevision
cd onevision/install
sh ./install.sh    # Create conda environment
sudo ./install.sh  # Install package using `sudo`. When prompt to input the 
                   # dataset directory path, you should enter: <some-path>/one/datasets

# Install `mish-cuda` package
cd mish-cuda
python setup.py build install
```


## <div align="center"> How To Use </div>
To be updated.


## <div align="center"> Citation </div>
If you find our work useful, please cite the following:

```text
@misc{Pham2022,  
    author = {Long Hoang Pham},  
    title = {OneVision: One Computer Vision Library To Rule Them All},  
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/phlong3105/onevision}},
    year = {2022},
}
```


## <div align="center"> Contact </div>
If you have any questions, feel free to contact `Long Pham` ([phlong@skku.edu](phlong@skku.edu))
