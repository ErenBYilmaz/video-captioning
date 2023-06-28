FROM vanallenlab/miniconda:3.9

# run with:
# docker build -t ${USER}_container_mgpt4 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --progress=plain . -f Dockerfile
# docker run -it --rm --gpus all -p 127.0.0.1:7860:7860/tcp --ipc=host -v $(pwd):/code ${USER}_container_mgpt4

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# otherwise some nividia stuff will not install properly
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# some useful tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-releaseinfo-change update && apt-get install curl git unzip graphviz nano git-lfs build-essential manpages-dev curl ffmpeg libsm6 libxext6 -y && git lfs install && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && apt-get install -y && git lfs install && rm -rf /var/lib/apt/lists/*
COPY environment.yml .
RUN conda install -n base conda=23.5
# RUN conda env update -n base --file environment.yml
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "minigpt4", "/bin/bash", "-c"]
RUN python -c "import scipy"
RUN conda init bash
RUN python -c "import scipy"
#RUN pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
#RUN pip install omegaconf torchvision iopath timm opencv-python webdataset decord accelerate scipy
#RUN pip install -i https://test.pypi.org/simple/ bitsandbytes==0.37.0
# RUN pip uninstall -y pytorch torchvision bitsandbytes
# RUN pip install pytorch torchvision
# RUN pip install --upgrade bitsandbytes==0.37.0
COPY main.py /opt/conda/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py
COPY main.py /opt/conda/envs/minigpt4/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py
# RUN conda install cudatoolkit=11.3 -y -c pytorch

# create current user inside image, so that newly created files belong to us and not to root and can be accessed outside docker as well
#ARG USER_ID
#ARG GROUP_ID
#RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && usermod -a -G root user && addgroup --gid 1011 data-acc && usermod -a -G data-acc user
#USER user
#WORKDIR /home/user

# Create environment
#RUN pip install --upgrade cmake
#RUN pip install hanging_threads tensorflow-addons==0.13.0 colorama matplotlib numpy pandas cachetools tabulate yappi joblib scipy gitpython markdown2 seaborn psutil pydot numba scikit-learn scikit-image pygments natsort voluptuous humanfriendly coloredlogs simpleitk itk==5.0.1 openpyxl jupyterlab "xlrd < 2" pydot tensorflow_probability==0.13.0 pydicom click sympy lifelines dill tensorboard_plugin_profile plotly wandb cmake==3.17.1
## outside of docker you also need tensorflow==2.5.0 and protobuf==3.20
#RUN pip install MulticoreTSNE

# RUN pip install fsspec omegaconf --upgrade

WORKDIR /code
