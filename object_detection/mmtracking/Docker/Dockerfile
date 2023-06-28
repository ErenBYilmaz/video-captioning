FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# run with:
# docker build -t ${USER}_container_tf2 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -f DockerfileTF25
# docker run -it --rm --gpus all --user "$(id -u):$(id -g)" --ipc=host -v $(pwd):/code ${USER}_container_tf2

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# otherwise some nividia stuff will not install properly
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# some useful tools
#RUN apt-get update && apt-get install curl git unzip graphviz nano -y && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install curl git unzip graphviz nano -y


# create current user inside image, so that newly created files belong to us and not to root and can be accessed outside docker as well
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && usermod -a -G root user && addgroup --gid 1011 data-acc && usermod -a -G data-acc user
USER user
# WORKDIR /home/user
WORKDIR /code/mmtracking
# Create environment
# RUN python -m venv open-mmlab
# RUN source open-mmlab/bin/activate
#  (optional)
# RUN pip install git+https://github.com/votchallenge/toolkit.git
# install the latest mmcv
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.1/index.html

# install mmdetection
RUN pip install mmdet

# install mmtracking
RUN git clone https://github.com/open-mmlab/mmtracking.git
RUN cd mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .
RUN pip install git+https://github.com/JonathonLuiten/TrackEval.git
RUN pip install git+https://github.com/lvis-dataset/lvis-api.git
RUN pip install git+https://github.com/TAO-Dataset/tao.git

WORKDIR /code