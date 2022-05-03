#!/bin/bash

# Install:
# chmod +x install.sh
# conda init bash
# ./install.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
onevision_dir=$(dirname "$current_dir")
one_dir=$(dirname "$onevision_dir")

# Add conda-forge channel
echo "Add 'conda-forge':"
conda config --append channels conda-forge

# Update 'base' env
echo "Update 'base' environment:"
conda update --a --y
pip install --upgrade pip


case "$OSTYPE" in
  linux*)
    echo "Linux / WSL"
    # Create `one` env
    echo "Create 'one' environment:"
    env_yml_path="${current_dir}/environment.yml"
    conda env create -f "${env_yml_path}"
    eval "$(conda shell.bash hook)"
    conda activate one
    pip install --upgrade pip
    # conda update --a --y
    # conda clean --a --y
    
    # Install `mish-cuda`
    echo "Install 'mish-cuda':"
    pip install git+https://github.com/JunnYu/mish-cuda.git
    ;;
  darwin*)
    echo "Mac OS"
    # Create `one` env
    echo "Create 'one' environment:"
    env_yml_path="${current_dir}/environment_macos.yml"
    conda env create -f "${env_yml_path}"
    conda activate one
    pip install --upgrade pip
    conda update --a --y
    conda clean --a --y

    # Install from 'brew'
    echo "Install from 'brew':"
    brew install ffmpeg
    brew install libvips
    ;;
  win*)
    echo "Windows"
    ;;
  msys*)
    echo "MSYS / MinGW / Git Bash"
    ;;
  cygwin*)
    echo "Cygwin"
    ;;
  bsd*)
    echo "BSD"
     ;;
  solaris*)
    echo "Solaris"
    ;;
  *)
    echo "unknown: $OSTYPE"
    ;;
esac


# Set environment variables
# shellcheck disable=SC2162
datasets_dir="${one_dir}/datasets"
read -e -i "$datasets_dir" -p "Enter DATASETS_DIR= " input
datasets_dir="${input:-$datasets_dir}"

if [ "$datasets_dir" != "" ]; then
  export DATASETS_DIR="$datasets_dir"
  conda env config vars set datasets_dir="$datasets_dir"
  echo "DATASETS_DIR has been set."
else
  echo "DATASETS_DIR has NOT been set."
fi

if [ -d "$onevision_dir" ];
then
	echo "DATASETS_DIR=$datasets_dir" > "${onevision_dir}/pycharm.env"
fi
