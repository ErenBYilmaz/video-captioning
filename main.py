from video_to_img.youtube_to_img import YoutubeToImageConverter

from img_to_text.minigpt4_assisted_gpt3 import MiniGPT4AssistedGPT3
from text_to_srt.text_to_srt_converter import SRTCreator


def main():
    # MiniGPT4AssistedGPT3().clear_output_cache()
    pipeline = SRTCreator(YoutubeToImageConverter('https://www.youtube.com/watch?v=uQITWbAaDx0'), MiniGPT4AssistedGPT3())
    # pipeline = TextToSRTConverter(ExampleVideoToImageConverter(), MiniGPT4AssistedGPT3())
    pipeline.create_srt()


if __name__ == '__main__':
    main()
