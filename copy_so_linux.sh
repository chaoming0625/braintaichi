#! /bin/sh

pip install taichi
chmod +x ./copy_so_linux.py
python copy_so_linux.py

# Ensure the set_env.sh script is actually created and is not empty
dir /project
if [ -s set_env.sh ]; then
    source /project/set_env.sh
    cp "$taichi_runtime_lib_dir"/libtaichi_c_api.so /project/brainpylib/
else
    echo "Environment setup script 'set_env.sh' is missing or empty."
    exit 1
fi
