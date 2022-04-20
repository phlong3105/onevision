#!/bin/bash

# Absolute path to this script:
script_path=$(readlink -f "$0")
current_dir=$(dirname "$script_path")
onevision_dir=$(dirname "$current_dir")

conda activate one

# Set environment variables
# shellcheck disable=SC2162
read -p "Enter DATASETS_DIR=" datasets_dir
if [ "$datasets_dir" != "" ]; then
  export DATASET_DIR="$datasets_dir"
  conda env config vars set datasets_dir="$datasets_dir" 
  echo "DATASETS_DIR has been set."
else
  echo "DATASETS_DIR has NOT been set."
fi

if [ -d "$onevision_dir" ];
then
	echo "DATASETS_DIR=$datasets_dir" > "${onevision_dir}/pycharm.env"
fi
