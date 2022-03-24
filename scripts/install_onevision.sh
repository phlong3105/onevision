#!/bin/bash

conda activate one

# Absolute path to `one` dir:
script_path=$(readlink -f "$0")       # workspaces/one/onevision/scripts/install_onecv.sh
dirpath=$(dirname "$script_path")     # workspaces/one/onevision/scripts
onevision_dir=$(dirname "$dirpath")   # workspaces/one/onevision
one_dir=$(dirname "$dirpath")         # workspaces/one

# Install upgrade `onevision`
echo "Install upgrade 'onevision'"
cd "$onevision_dir" || exit
pip install ./ --upgrade

conda update --a --y
conda clean --a --y
