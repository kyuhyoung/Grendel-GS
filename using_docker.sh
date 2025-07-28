#docker_name=u_22_cpp_11_3_py_3_10_cv_4_7
#docker_name=align
docker_name=bmseg
dir_cur=/workspace/${PWD##*/}
#dir_data=/mnt/hdd_16tb/dataset_stereo/zy3_sat_stereo_image/Sainte-Maxime/
#dir_data=/home/kevin-sosa/work/etc/gwarp_pp_docker/data
#dir_data=/mnt/hdd_16tb/dataset_stereo/dabeeo
dir_data=/raid/HDD/dataset_stereo

####################################################################################
#: << 'END'
#   docker build
docker buildx build --platform linux/amd64 --force-rm --shm-size=64g -t ${docker_name} -f docker_file/Dockerfile_${docker_name} .
#END

#: << 'END'
#   docker info.
#docker run --rm -it -w $PWD -v $PWD:$PWD ${docker_name} bash docker_file/extract_docker_info.sh
docker run --platform linux/amd64 --rm -it -w $PWD -v $PWD:$PWD ${docker_name} sh -c ". ~/.bashrc && . ./extract_docker_info.sh"
#END

#   docker run
##	for SSH remote docker
#docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} /bin/bash
docker run -p 8080:8080 --rm -it --shm-size=64g --gpus device=0 -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY --privileged -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} bash
#docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} sh -c "bash usage.sh"
##	for local docker
# xhost +local:docker # in another terminal
#export DISPLAY=:0 && docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix ${docker_name} fish



#END
