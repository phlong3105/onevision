#!/bin/bash

# Install: 
# chmod +x install_env_mac.sh
# ./install_env_mac.sh

script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")

# Add conda-forge channel
conda config --append channels conda-forge

# Update 'base' env
echo "Update base environment:"
conda update --a --y
pip install --upgrade pip
brew install ffmpeg
brew install libvips

# Create `one` env
echo "Create 'one' environment:"
env_yml_path="${current_dir}/environment_macos.yml"
conda env create -f "${env_yml_path}"
conda activate one
conda update --a --y
pip install --upgrade pip 
conda clean --a --y

# Setup system environment variables
set_environ_script="${current_dir}/set_environ.sh"
"$set_environ_script"
