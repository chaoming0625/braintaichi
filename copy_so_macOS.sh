#! /bin/sh

pip install taichi==1.7.2
chmod +x ./copy_so_macOS.py
python copy_so_macOS.py

# Ensure the set_env.sh script is actually created and is not empty
# Get the absolute path of the project/braintaichi directory
project_path="$(dirname "$(realpath "$0")")/project/braintaichi"

# Set the absolute path as an environment variable
export PROJECT_PATH="$project_path"

# Source the set_env.sh script
source "$project_path/set_env.sh"

# Copy the libtaichi_c_api.dylib file to the project/braintaichi directory
cp "$taichi_runtime_lib_dir"/libtaichi_c_api.dylib "$project_path/"

# List the contents of the project/braintaichi directory
ls "$project_path"

