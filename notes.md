# Outline

- [ ] technische Schnittstelle zum Sprachmodell
  - [ ] Auswahl des Modells
- [ ] Prompt-Engineering
- [x] Videoauswahl für Entwicklung und Tests
  - [x] evtl erstmal testbilder
- [x] Bilder aus Videos ausschneiden
- [ ] Design: Wie kann das nachher gut aussehen? Details/Mockup bzgl. Overlay
- [ ] Technische Umsetzung: Einbetten ins Video
- [ ] Videos anschauen und nach Fehlern/Verbesserungsmöglichkeiten schauen

# Additional ideas
- [ ] provide more context to the model in text form?
- [ ] haystack?
- [ ] Kombination von Modellen? MiniGPT-4 -> GPT 3.5-Turbo

# Schnittstellen
- [x] Bilddatei im png/jpg format, dazu zu jedem Bild eine json datei mit gleichem namen aber anderer endung
  - [ ] Die JSON-datei enthält metadaten inklusive out des tools
  - [x] bilddateien nach zeit organisieren (im dateinamen sodass man danach sortieren kann)
- [ ] Eine Logdatei die die verarbeitungsschritte trackt (bilddübergreifend)
- Im wesentlichen drei komponenten:
  1. Bild inklusive Metadaten aus Video extrahieren
  2. Bild zu Text
  3. Text anzeigen

# Video libraries
 - moviepy: https://zulko.github.io/moviepy/examples/painting_effect.html
 - opencv: https://pypi.org/project/cv2watermark/
 - ffmpeg?


# Useful links:
- https://zulko.github.io/moviepy/examples/painting_effect.html
- https://www.eventbrite.de/e/codingwaterkant-2023-tickets-493782716397
- https://www.waterkant.sh/de/coding-waterkant

## Models
- https://huggingface.co/spaces/Vision-CAIR/minigpt4
  - https://github.com/Vision-CAIR/MiniGPT-4
  - https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md
    - https://huggingface.co/docs/transformers/main/model_doc/llama
    - https://github.com/Vision-CAIR/MiniGPT-4/issues/59#issuecomment-1515129713
- https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA

# MiniGPT-4 weight-merge command history
```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0
git clone https://huggingface.co/decapoda-research/llama-13b-hf
python -m fastchat.model.apply_delta --base /code/vicuna_weights/llama-13b-hf/  --target /code/vicuna_weights/merged/  --delta /code/vicuna_weights/vicuna-13b-delta-v0/
```

# MiniGPT-4 dockerfile
```dockerfile
FROM continuumio/anaconda3:latest

# run with:
# docker build -t ${USER}_container_mgpt4 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . -f Dockerfile
# docker run -it --rm --gpus all --user "$(id -u):$(id -g)" --ipc=host -v $(pwd):/code ${USER}_container_mgpt4

ENV PYTHONPATH="/code/"

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# otherwise some nividia stuff will not install properly
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# some useful tools
RUN apt-get --allow-releaseinfo-change update && apt-get install curl git unzip graphviz nano git-lfs build-essential manpages-dev -y && git lfs install && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda update -n base -c defaults conda && conda env create -f environment.yml
SHELL ["conda", "run", "-n", "minigpt4", "/bin/bash", "-c"]

# create current user inside image, so that newly created files belong to us and not to root and can be accessed outside docker as well
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && usermod -a -G root user && addgroup --gid 1011 data-acc && usermod -a -G data-acc user
USER user
WORKDIR /home/user

# Create environment
#RUN pip install --upgrade cmake
#RUN pip install hanging_threads tensorflow-addons==0.13.0 colorama matplotlib numpy pandas cachetools tabulate yappi joblib scipy gitpython markdown2 seaborn psutil pydot numba scikit-learn scikit-image pygments natsort voluptuous humanfriendly coloredlogs simpleitk itk==5.0.1 openpyxl jupyterlab "xlrd < 2" pydot tensorflow_probability==0.13.0 pydicom click sympy lifelines dill tensorboard_plugin_profile plotly wandb cmake==3.17.1
## outside of docker you also need tensorflow==2.5.0 and protobuf==3.20
#RUN pip install MulticoreTSNE

RUN pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10

WORKDIR /code
```