#!/bin/bash

# Install: 
# chmod +x install_env_linux.sh
# ./install_env_linux.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")

# Add conda-forge channel
echo "Add 'conda-forge':"
conda config --append channels conda-forge

# Update 'base' env
echo "Update 'base' environment:"
conda update --a --y
pip install --upgrade pip 
apt-get install libvips -y

# Create `one` env
echo "Create 'one' environment:"
env_yml_path="${current_dir}/environment.yml"
conda env create -f "${env_yml_path}"
conda activate one
conda update --a --y
pip install --upgrade pip
conda clean --a --y

# Install 'cuda' and 'cudnn'
echo "Install 'cuda':"
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
apt-get update
apt-get install cuda-11-3
apt-get install libcudnn8
apt-get install ffmpeg

# Install `mish-cuda`
echo "Install 'mish-cuda':"
mish_cuda_dir="${current_dir}/mish-cuda"
cd "$mish_cuda_dir" || exit
apt install gcc-10 -y
apt install g++-10 -y
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
python setup.py build install

# Setup system environment variables
set_environ_script="${current_dir}/set_environ.sh"
"$set_environ_script"
