docker_name=ogs
dir_cur=/workspace/${PWD##*/}
#dir_data=/mnt/hdd_16tb/dataset_stereo
dir_data=/media2/data/dataset_stereo/non-sat
log_file="using_docker.log"

##########################################################################################
#   docker build (skip with -nb option)
if [ "$1" != "-nb" ]; then
    sudo docker buildx build --platform linux/amd64 --force-rm --shm-size=64g -t ${docker_name} -f docker_file/Dockerfile_${docker_name} . 2>&1 | tee ${log_file}
fi

#   docker info. (skip with -nb option)
if [ "$1" != "-nb" ]; then
    sudo docker run --platform linux/amd64 --rm -it -w $PWD -v $PWD:$PWD ${docker_name} sh -c ". ~/.bashrc && . ./extract_docker_info.sh" 2>&1 | tee -a ${log_file}
fi

#   docker run (no logging inside container)
##	for SSH remote docker
if [ -z "$*" ] || { [ "$1" = "-nb" ] && [ -z "$2" ]; }; then
    # No command given → interactive shell
    sudo docker run --rm -it --shm-size=64g --gpus '"device=6,7"' -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} bash
else
    # Run given command
    cmd="$*"
    [ "$1" = "-nb" ] && shift && cmd="$*"
    sudo docker run --rm -it --shm-size=64g --gpus '"device=6,7"' -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} bash -c "cd ${dir_cur} && ${cmd}"
fi

##	for local docker
# xhost +local:docker # in another terminal
#export DISPLAY=:0 && docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix ${docker_name} fish
