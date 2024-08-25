#!/bin/bash

# Get the current Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")

# Set the CIBW_BUILD environment variable based on the Python version
if [[ $python_version == "3.8" ]]; then
    export CIBW_BUILD="cp38-macosx_x86_64"
elif [[ $python_version == "3.9" ]]; then
    export CIBW_BUILD="cp39-macosx_x86_64*"
elif [[ $python_version == "3.10" ]]; then
    export CIBW_BUILD="cp310-macosx_x86_64*"
elif [[ $python_version == "3.11"]]; then
    export CIBW_BUILD="cp311-macosx_x86_64"
elif [[ $python_version == "3.12"]]; then
    export CIBW_BUILD="cp312-macosx_x86_64"
else
    echo "Unsupported Python version: $python_version"
    exit 1
fi