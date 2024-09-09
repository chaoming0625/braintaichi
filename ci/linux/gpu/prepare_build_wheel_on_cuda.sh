#! /bin/sh

set -e
set -x

#pip install "cmake==3.22.*"
rm setup.py
mv setup_cuda.py setup.py
# mv setup_cpu.py setup.py

# https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/

# Install CUDA 11.2, see:
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
# * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
#yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
#yum install --setopt=obsoletes=0 -y \
#    cuda-nvcc-11-2-11.2.152-1 \
#    cuda-cudart-devel-11-2-11.2.152-1 \
#    libcurand-devel-11-2-10.2.3.152-1 \
#    libcudnn8-devel-8.1.1.33-1.cuda11.2 \
#    libcublas-devel-11-2-11.4.1.1043-1
#ln -s cuda-11.2 /usr/local/cuda
#
#yum install -y python3-devel.x86_64


# Install CUDA 11.4
#yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
#yum install --setopt=obsoletes=0 -y \
#    cuda-nvcc-11-4-11.4.152-1 \
#    cuda-cudart-devel-11-4-11.4.148-1 \
#    libcurand-devel-11-4-10.2.5.120-1 \
#    libcudnn8-devel-8.2.4.15-1.cuda11.4 \
#    libcublas-devel-11-4-11.6.5.2-1
#ln -s cuda-11.4 /usr/local/cuda
#
#yum install -y python3-devel.x86_64


# Install CUDA 11.6
# sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf
#yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
#yum install --setopt=obsoletes=0 -y \
#    cuda-nvcc-11-6-11.6.124-1 \
#    cuda-cudart-devel-11-6-11.6.55-1 \
#    libcurand-devel-11-6-10.2.9.124-1 \
#    libcudnn8-devel-8.4.1.50-1.cuda11.6 \
#    libcublas-devel-11-6-11.9.2.110-1
#ln -s cuda-11.6 /usr/local/cuda


# Install CUDA 11.8
# yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
# yum install --setopt=obsoletes=0 -y \
#    cuda-nvcc-11-8-11.8.89-1 \
#    cuda-cudart-devel-11-8-11.8.89-1 \
#    libcurand-devel-11-8-10.3.0.86-1\
#    libcudnn8-devel-8.8.0.121-1.cuda11.8 \
#    libcublas-devel-11-8-11.11.3.6-1
# ln -s cuda-11.8 /usr/local/cuda

# yum install -y python3-devel.x86_64

# Install CUDA 12.0
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install --setopt=obsoletes=0 -y \
   cuda-drivers-525.147.05-1.x86_64 \
   cuda-nvcc-12-0-12.0.140-1 \
   cuda-cudart-devel-12-0-12.0.146-1 
ln -s cuda-12.0 /usr/local/cuda

yum install glibc-devel

yum install --disableplugin=fastmirror -y python3-devel.x86_64

pip_config_dir="${HOME}/.pip"
pip_config_file="${pip_config_dir}/pip.conf"

mkdir -p "${pip_config_dir}"

if [[ ! -f "${pip_config_file}" ]]; then
    touch "${pip_config_file}"
fi

echo "[global]" >> "${pip_config_file}"
# echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> "${pip_config_file}"
echo "index-url = https://pypi.org/simple" >> "${pip_config_file}"

echo "Pip mirror sources have been set!"