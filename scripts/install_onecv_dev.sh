#!/bin/bash

conda activate one

# Absolute path to `one` dir:
script_path=$(readlink -f "$0")     # workspaces/one/onecv/scripts/install_onecv_dev.sh
dirpath=$(dirname "$script_path")   # workspaces/one/onecv/scripts/
dirpath=$(dirname "$dirpath")       # workspaces/one/onecv/
one=$(dirname "$dirpath")           # workspaces/one/

# Install upgrade `onecv`
echo "Install upgrade 'onecv'"
onecv="${one}/onecv"  # workspaces/one/onecv
cd "$onecv" || exit
pip install -e ./ --upgrade

conda update --a --y
conda clean --a --y
