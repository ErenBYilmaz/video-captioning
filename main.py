from img_to_text.minigpt4 import MiniGPT4Captioning
from img_to_text.minigpt4_assisted_gpt3 import MiniGPT4AssistedGPT3
from text_to_srt.text_to_srt_converter import TextToSRTConverter
from video_to_img.example_video_converter import ExampleVideoToImageConverter


def main():
    # MiniGPT4AssistedGPT3().clear_output_cache()
    pipeline = TextToSRTConverter(ExampleVideoToImageConverter(), MiniGPT4AssistedGPT3())
    pipeline.create_srt()


if __name__ == '__main__':
    main()
