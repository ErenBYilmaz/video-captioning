# Outline

- [x] technische Schnittstelle zum Sprachmodell
  - [x] Auswahl des Modells
- [ ] Prompt-Engineering
- [x] Videoauswahl für Entwicklung und Tests
  - [x] evtl erstmal testbilder
- [x] Bilder aus Videos ausschneiden
- [ ] Design: Wie kann das nachher gut aussehen? Details/Mockup bzgl. Overlay
- [x] Technische Umsetzung: Untertiteldatei
- [ ] Technische Umsetzung: Einbetten ins Video
- [ ] Videos anschauen und nach Fehlern/Verbesserungsmöglichkeiten schauen

# Additional ideas
- [ ] provide more context to the model in text form?
- [ ] haystack?
- [ ] Kombination von Modellen? MiniGPT-4 -> GPT 3.5-Turbo

# Schnittstellen
- [x] Bilddatei im png/jpg format, dazu zu jedem Bild eine json datei mit gleichem namen aber anderer endung
  - [x] Die JSON-datei enthält metadaten inklusive out des tools
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
  - https://github.com/TimDettmers/bitsandbytes/issues/156
  - https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
- https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA

# MiniGPT-4 weight-merge command history
```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0
git clone https://huggingface.co/decapoda-research/llama-13b-hf
python -m fastchat.model.apply_delta --base /code/vicuna_weights/llama-13b-hf/  --target /code/vicuna_weights/merged/  --delta /code/vicuna_weights/vicuna-13b-delta-v0/
conda activate minigpt4
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 7
```
