FROM vanallenlab/miniconda:3.9

# run with:
# docker build -t ${USER}_container_mgpt4 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --progress=plain . -f Dockerfile
# docker run -it --rm --gpus all -p 127.0.0.1:7860:7860/tcp --ipc=host -v $(pwd):/code ${USER}_container_mgpt4

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# some useful tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-releaseinfo-change update && apt-get install curl git unzip graphviz nano git-lfs build-essential manpages-dev curl ffmpeg libsm6 libxext6 -y && git lfs install && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && apt-get install -y && git lfs install && rm -rf /var/lib/apt/lists/*
COPY environment.yml .
RUN conda install -n base conda=23.5
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "minigpt4", "/bin/bash", "-c"]
RUN conda init bash
COPY main.py /opt/conda/envs/minigpt4/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py

WORKDIR /code
