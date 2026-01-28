docker_name=ogs
dir_cur=/workspace/${PWD##*/}
#dir_data=/mnt/hdd_16tb/dataset_stereo
dir_data=/media2/data/dataset_stereo
log_file="using_docker.log"

##########################################################################################
#   docker build (skip with -nb option)
if [ "$1" != "-nb" ]; then
    docker buildx build --platform linux/amd64 --force-rm --shm-size=64g -t ${docker_name} -f docker_file/Dockerfile_${docker_name} . 2>&1 | tee ${log_file}
fi

#   docker info.
docker run --platform linux/amd64 --rm -it -w $PWD -v $PWD:$PWD ${docker_name} sh -c ". ~/.bashrc && . ./extract_docker_info.sh" 2>&1 | tee -a ${log_file}

#   docker run (no logging inside container)
##	for SSH remote docker
docker run --rm -it --shm-size=64g --gpus device=0 -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY --privileged -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} bash

##	for local docker
# xhost +local:docker # in another terminal
#export DISPLAY=:0 && docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix ${docker_name} fish
