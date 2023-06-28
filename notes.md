# Outline

- [ ] technische Schnittstelle zum Sprachmodell
  - [ ] Auswahl des Modells
- [ ] Prompt-Engineering
- [ ] Videoauswahl für Entwicklung und Tests
  - [ ] evtl erstmal testbilder
- [ ] Bilder aus Videos ausschneiden
- [ ] Design: Wie kann das nachher gut aussehen? Details/Mockup bzgl. Overlay
- [ ] Technische Umsetzung: Einbetten ins Video
- [ ] Videos anschauen und nach Fehlern/Verbesserungsmöglichkeiten schauen

# Additional ideas
- [ ] provide more context to the model in text form?
- [ ] haystack?
- [ ] Kombination von Modellen? MiniGPT-4 -> GPT 3.5-Turbo

# Schnittstellen
- [ ] Bilddatei im png format, dazu zu jedem Bild eine json datei mit gleichem namen aber anderer endung
  - [ ] Die JSON-datei enthält metadaten inklusive out des tools
  - [ ] bilddateien nach zeit organisieren (im dateinamen sodass man danach sortieren kann)
- [ ] Eine Logdatei die die verarbeitungsschritte trackt (bilddübergreifend)
- Im wesentlichen drei komponenten:
  1. Bild inklusive Metadaten aus Video extrahieren
  2. Bild zu Text
  3. Text anzeigen

# Video libraries
 - moviepy: https://zulko.github.io/moviepy/examples/painting_effect.html
 - opencv
 - ffmpeg?

# Useful links:
- https://zulko.github.io/moviepy/examples/painting_effect.html
- https://huggingface.co/spaces/Vision-CAIR/minigpt4
  - https://github.com/Vision-CAIR/MiniGPT-4
  - https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md
    - https://huggingface.co/docs/transformers/main/model_doc/llama
- https://www.eventbrite.de/e/codingwaterkant-2023-tickets-493782716397
- https://www.waterkant.sh/de/coding-waterkant