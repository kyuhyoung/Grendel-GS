#!/bin/bash
echo "==================================================================================================="
echo "Docker container info."
echo "==================================================================================================="
echo "USAGE :"
echo " docker build --force-rm -t \${docker_name} -f docker_file/Dockerfile_\${docker_name}"
echo " docker run -w \$PWD -v \$PWD:\$PWD \${docker_name} sh -c \". ~/.bashrc && . ./extract_docker_info.sh\""
echo "---------------------------------------------------------------------------------------------------"
os=$(cat /etc/os-release | grep 'PRETTY_NAME' | cut -d '=' -f 2 | cut -d '"' -f 2)
echo "OS : ${os}"
cpp=$(g++ --version 2>/dev/null | grep 'g++' | rev | cut -d' ' -f1 | rev)
echo "c++ : ${cpp}"
python=$(python3 --version | xargs | rev | cut -d' ' -f1 | rev)
echo "python : ${python}"
cuda=$(ls -l /usr/local | grep "cuda ->" | rev | cut -d' ' -f1 | rev | xargs realpath | rev | cut -d'-' -f1 | rev)
echo "cuda : ${cuda}"
cudnn1=$(find /usr/lib/x86_64-linux-gnu -name "libcudnn.so.*.*" | rev | cut -d'.' -f3 | rev)
cudnn2=$(find /usr/lib/x86_64-linux-gnu -name "libcudnn.so.*.*" | rev | cut -d'.' -f2 | rev)
cudnn3=$(find /usr/lib/x86_64-linux-gnu -name "libcudnn.so.*.*" | rev | cut -d'.' -f1 | rev)
echo "cudnn : ${cudnn1}.${cudnn2}.${cudnn3}"
torch=$(pip3 list | grep "torch " | xargs | rev | cut -d' ' -f1 | rev)
echo "pytorch : ${torch}"
tv=$(pip3 list | grep torchvision | xargs | rev | cut -d' ' -f1 | rev)
echo "torchvison : ${tv}"
gsplat=$(pip3 list | grep gsplat | xargs | rev | cut -d' ' -f1 | rev)
echo "gsplat : ${gsplat}"
pytorch3d=$(pip3 list | grep pytorch3d | xargs | rev | cut -d' ' -f1 | rev)
echo "pytorch3d : ${pytorch3d}"
tb=$(pip3 list | grep "tensorboard " | xargs | rev | cut -d' ' -f1 | rev)
echo "tensorboard : ${tb}"
echo "==================================================================================================="
