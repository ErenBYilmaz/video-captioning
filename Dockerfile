FROM python:3.11.4

# run with:
# docker build -t ${USER}_video_captioning --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -f Dockerfile
# docker run -it --rm --gpus all --user "$(id -u):$(id -g)" --ipc=host -v $(pwd):/code ${USER}_video_captioning

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# some useful tools
RUN apt-get update && apt-get install curl git unzip graphviz nano ffmpeg libsm6 libxext6 -y && rm -rf /var/lib/apt/lists/*

# create current user inside image, so that newly created files belong to us and not to root and can be accessed outside docker as well
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && usermod -a -G root user && addgroup --gid 1011 data-acc && usermod -a -G data-acc user
USER user
WORKDIR /home/user

# Create environment
RUN pip install --upgrade cmake pip setuptools wheel
RUN pip install matplotlib numpy pandas scipy tabulate cachetools pygments websocket-client opencv-python pysrt requests cv2watermark sopenai



WORKDIR /code